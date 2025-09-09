# -*- coding: utf-8 -*-
import os, time, json, hmac, hashlib, math, threading, sqlite3, urllib.parse
from datetime import datetime, timedelta
from collections import deque
from decimal import Decimal, ROUND_DOWN

import requests
from flask import Flask, jsonify, send_from_directory
import random
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============== Binance API ==============
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
    raise ValueError("Please set BINANCE_API_KEY and BINANCE_SECRET_KEY in your environment")

FAPI = "https://fapi.binance.com"
SESSION = requests.Session()
TIMEOUT = 10
_server_offset_ms = 0

def sync_server_time():
    global _server_offset_ms
    r = SESSION.get(f"{FAPI}/fapi/v1/time", timeout=TIMEOUT)
    r.raise_for_status()
    server_ms = r.json()["serverTime"]
    local_ms = int(time.time() * 1000)
    _server_offset_ms = server_ms - local_ms

def ts_ms():
    return int(time.time() * 1000 + _server_offset_ms)

def get_binance_signature(query_string: str) -> str:
    return hmac.new(
        BINANCE_SECRET_KEY.encode("utf-8"),
        query_string.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

def binance_request(endpoint, method="GET", params=None, signed=False, recvWindow=5000):
    if params is None:
        params = {}
    headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
    if signed:
        params["timestamp"] = ts_ms()
        params["recvWindow"] = recvWindow
        query = urllib.parse.urlencode(params, doseq=True)
        params["signature"] = get_binance_signature(query)
    url = f"{FAPI}{endpoint}"

    for attempt in range(3):
        try:
            resp = SESSION.request(method, url, headers=headers, params=params, timeout=TIMEOUT)
            if resp.status_code in (418, 429):
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt == 2:
                raise
            sync_server_time()
            time.sleep(2 ** attempt)

# ============== Exchange Info Cache ==============
_symbol_info = {}

def load_exchange_info():
    data = binance_request("/fapi/v1/exchangeInfo", "GET", signed=False)
    for s in data["symbols"]:
        sym = s["symbol"]
        pf = next(f for f in s["filters"] if f["filterType"] == "PRICE_FILTER")
        lf = next(f for f in s["filters"] if f["filterType"] in ("MARKET_LOT_SIZE", "LOT_SIZE"))
        mn_filter = next((f for f in s["filters"] if f["filterType"] == "MIN_NOTIONAL"), None)
        _symbol_info[sym] = {
            "tickSize": Decimal(pf["tickSize"]),
            "stepSize": Decimal(lf["stepSize"]),
            "minNotional": Decimal(mn_filter["notional"]) if mn_filter else None
        }

def q_price(sym, price_float):
    info = _symbol_info[sym]
    step = info["tickSize"]
    val = (Decimal(str(price_float)) / step).to_integral_value(ROUND_DOWN) * step
    return float(val)

def q_qty(sym, qty_float):
    info = _symbol_info[sym]
    step = info["stepSize"]
    val = (Decimal(str(qty_float)) / step).to_integral_value(ROUND_DOWN) * step
    return float(val)

def meets_min_notional(sym, price_float, qty_float):
    info = _symbol_info.get(sym, {})
    min_notional = info.get("minNotional")
    if min_notional is None:
        min_notional = Decimal("5")
    notional = Decimal(str(price_float)) * Decimal(str(qty_float))
    return notional >= min_notional

# ============== Configuration ==============
QUICK_CHECK_INTERVAL = 15
MAX_CONCURRENT_POSITIONS = 8
MIN_CONFIDENCE_THRESHOLD = 6

POSITION_SCALING = {
    0:0.20, 1:0.17, 2:0.14, 3:0.12,
    4:0.10, 5:0.08, 6:0.07, 7:0.06
}

portfolio = {
    "balance": 0.0,
    "positions": {},
    "trade_history": [],
    "learning_data": {"successful_patterns":[], "failed_patterns":[], "performance_metrics":{}, "pattern_memory":{}}
}

# ============== Database ==============
DB_PATH = "trading_bot.db"

def db_connect():
    conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    return conn

def db_execute(sql, params=(), many=False, fetch=False):
    for attempt in range(5):
        try:
            conn = db_connect()
            cur = conn.cursor()
            if many:
                cur.executemany(sql, params)
            else:
                cur.execute(sql, params)
            rows = cur.fetchall() if fetch else None
            conn.commit()
            conn.close()
            return rows
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and attempt < 4:
                time.sleep(0.25 * (attempt + 1))
                continue
            raise

def init_database():
    conn = db_connect()
    c = conn.cursor()
    c.execute("PRAGMA journal_mode=WAL;")
    c.execute("PRAGMA synchronous=NORMAL;")
    c.executescript("""
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        position_id TEXT UNIQUE,
        coin TEXT NOT NULL,
        action TEXT NOT NULL,
        direction TEXT,
        price REAL NOT NULL,
        position_size REAL,
        leverage INTEGER,
        notional_value REAL,
        stop_loss REAL,
        take_profit REAL,
        pnl REAL,
        pnl_percent REAL,
        duration TEXT,
        reason TEXT,
        confidence INTEGER,
        profitable BOOLEAN,
        market_conditions TEXT,
        timeframe_analysis TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS active_positions (
        position_id TEXT PRIMARY KEY,
        coin TEXT NOT NULL,
        symbol TEXT NOT NULL,
        direction TEXT NOT NULL,
        entry_price REAL NOT NULL,
        entry_time TEXT NOT NULL,
        position_size REAL NOT NULL,
        leverage INTEGER NOT NULL,
        notional_value REAL NOT NULL,
        stop_loss REAL NOT NULL,
        take_profit REAL NOT NULL,
        duration_target TEXT,
        confidence INTEGER,
        reasoning TEXT,
        market_regime TEXT,
        atr_at_entry REAL,
        last_checked DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS ai_decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        coin TEXT NOT NULL,
        direction TEXT NOT NULL,
        confidence INTEGER NOT NULL,
        reasoning TEXT,
        market_regime TEXT,
        context_used BOOLEAN DEFAULT TRUE,
        outcome_pnl REAL,
        ml_confidence REAL,
        pattern_strength REAL,
        regime_detected TEXT,
        features_hash TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS hyperparameters (
        key TEXT PRIMARY KEY,
        value REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS cost_tracking (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        service TEXT NOT NULL,
        cost REAL NOT NULL,
        units TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS daily_cost_summary (
        date DATE PRIMARY KEY,
        openai_cost REAL DEFAULT 0,
        api_calls INTEGER DEFAULT 0,
        trades_executed INTEGER DEFAULT 0,
        positions_checked INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS coin_stats (
        coin TEXT PRIMARY KEY,
        trades INTEGER DEFAULT 0,
        wins INTEGER DEFAULT 0,
        sum_pnl REAL DEFAULT 0,
        ema_wr REAL DEFAULT 0.5,
        ema_pnl REAL DEFAULT 0.0
    );
    CREATE TABLE IF NOT EXISTS ml_features (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        coin TEXT NOT NULL,
        features_json TEXT NOT NULL,
        outcome REAL,
        regime TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS pattern_memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pattern_hash TEXT NOT NULL,
        pattern_type TEXT NOT NULL,
        success_rate REAL DEFAULT 0.5,
        trade_count INTEGER DEFAULT 0,
        avg_pnl REAL DEFAULT 0.0,
        last_seen DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.commit()
    conn.close()
    print("📁 SQLite initialized (WAL) with ML extensions")

# ============== ML Feature Engineering ==============
class AdvancedFeatureExtractor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.pattern_memory = {}
        
    def extract_microstructure_features(self, orderbook_data, trade_flow=None):
        """Extract institutional footprints from market microstructure"""
        features = {}
        
        # Order book imbalance features
        bid_notional = orderbook_data.get("bid_notional", 0)
        ask_notional = orderbook_data.get("ask_notional", 0)
        total_liquidity = bid_notional + ask_notional
        
        features["orderbook_imbalance"] = (bid_notional - ask_notional) / max(total_liquidity, 1)
        features["liquidity_depth"] = min(1.0, total_liquidity / 1_000_000)
        features["spread_pressure"] = orderbook_data.get("spread", 0) * 10000  # bps
        
        # Detect large block activity (proxy)
        if total_liquidity > 5_000_000:
            features["institutional_activity"] = min(1.0, total_liquidity / 20_000_000)
        else:
            features["institutional_activity"] = 0.0
            
        # Hidden liquidity estimation
        expected_depth = features["liquidity_depth"] * 1.5
        actual_depth = features["liquidity_depth"]
        features["hidden_liquidity"] = max(0, expected_depth - actual_depth)
        
        return features
        
    def extract_regime_features(self, multi_tf_data, current_price):
        """Advanced regime detection using multiple timeframes"""
        features = {}
        
        # Cross-timeframe momentum alignment
        momentum_scores = []
        volatility_scores = []
        
        for tf, data in multi_tf_data.items():
            if "closes" not in data or len(data["closes"]) < 20:
                continue
                
            closes = np.array(data["closes"])
            returns = np.diff(np.log(closes))
            
            # Momentum features
            momentum_scores.append(np.mean(returns[-5:]) / np.std(returns[-20:]))
            
            # Volatility clustering
            vol_recent = np.std(returns[-5:])
            vol_baseline = np.std(returns[-20:])
            volatility_scores.append(vol_recent / max(vol_baseline, 1e-6))
            
        features["cross_tf_momentum"] = np.mean(momentum_scores) if momentum_scores else 0
        features["volatility_regime"] = np.mean(volatility_scores) if volatility_scores else 1
        
        # Market structure features
        if "1h" in multi_tf_data and "closes" in multi_tf_data["1h"]:
            closes = np.array(multi_tf_data["1h"]["closes"])
            
            # Trend strength using multiple EMAs
            ema_fast = self._ema(closes, 12)
            ema_slow = self._ema(closes, 26)
            features["trend_strength"] = (ema_fast - ema_slow) / ema_slow if ema_slow != 0 else 0
            
            # Price relative to multi-timeframe pivots
            recent_high = np.max(closes[-20:])
            recent_low = np.min(closes[-20:])
            features["price_position"] = (current_price - recent_low) / max(recent_high - recent_low, 1)
            
        return features
        
    def extract_pattern_features(self, multi_tf_data, market_data):
        """Extract learned pattern features"""
        features = {}
        
        # Volume-price divergence detection
        if "1h" in multi_tf_data:
            data = multi_tf_data["1h"]
            if "closes" in data and "volumes" in data and len(data["closes"]) >= 20:
                closes = np.array(data["closes"])
                volumes = np.array(data["volumes"])
                
                # Price momentum vs volume momentum
                price_mom = (closes[-5:].mean() - closes[-10:-5].mean()) / closes[-10:-5].mean()
                vol_mom = (volumes[-5:].mean() - volumes[-10:-5].mean()) / volumes[-10:-5].mean()
                features["pv_divergence"] = abs(price_mom - vol_mom)
                
                # Accumulation/Distribution approximation
                money_flow = closes * volumes
                features["money_flow_trend"] = (money_flow[-5:].mean() - money_flow[-10:-5].mean()) / money_flow[-10:-5].mean()
        
        # Fractal pattern recognition (simplified)
        features["fractal_dimension"] = self._estimate_fractal_dimension(multi_tf_data)
        
        return features
        
    def _ema(self, data, period):
        """Calculate EMA"""
        alpha = 2.0 / (period + 1.0)
        ema = data[0]
        for price in data[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema
        
    def _estimate_fractal_dimension(self, multi_tf_data):
        """Simplified fractal dimension for market complexity"""
        if "1h" not in multi_tf_data or "closes" not in multi_tf_data["1h"]:
            return 1.5  # neutral
            
        closes = np.array(multi_tf_data["1h"]["closes"][-30:])
        if len(closes) < 10:
            return 1.5
            
        # Simple box-counting approximation
        price_range = np.max(closes) - np.min(closes)
        if price_range == 0:
            return 1.0
            
        # Measure path length vs displacement
        total_distance = np.sum(np.abs(np.diff(closes)))
        direct_distance = abs(closes[-1] - closes[0])
        
        if direct_distance == 0:
            return 2.0  # maximum complexity
            
        complexity = total_distance / direct_distance
        return min(2.0, 1.0 + np.log(complexity) / 10)  # normalize to [1, 2]

# ============== Online Learning System ==============
class AdaptiveMLModel:
    def __init__(self):
        self.feature_buffer = deque(maxlen=1000)  # Keep recent features
        self.outcome_buffer = deque(maxlen=1000)  # Keep recent outcomes
        self.regime_models = {
            'trending_up': {'weights': np.random.randn(20) * 0.01, 'bias': 0.0},
            'trending_down': {'weights': np.random.randn(20) * 0.01, 'bias': 0.0},
            'ranging': {'weights': np.random.randn(20) * 0.01, 'bias': 0.0},
            'volatile': {'weights': np.random.randn(20) * 0.01, 'bias': 0.0}
        }
        self.learning_rate = 0.001
        self.model_health = "HEALTHY"
        self.feature_importance = np.ones(20) / 20
        
    def extract_features_vector(self, market_data, multi_tf_data, orderbook_data):
        """Convert market data to ML feature vector"""
        extractor = AdvancedFeatureExtractor()
        
        # Get all feature groups
        microstructure = extractor.extract_microstructure_features(orderbook_data)
        regime = extractor.extract_regime_features(multi_tf_data, market_data.get("price", 0))
        patterns = extractor.extract_pattern_features(multi_tf_data, market_data)
        
        # Combine into fixed-size vector (20 features)
        features = np.zeros(20)
        
        # Microstructure features (0-4)
        features[0] = microstructure.get("orderbook_imbalance", 0)
        features[1] = microstructure.get("liquidity_depth", 0.5)
        features[2] = microstructure.get("spread_pressure", 0)
        features[3] = microstructure.get("institutional_activity", 0)
        features[4] = microstructure.get("hidden_liquidity", 0)
        
        # Regime features (5-9)
        features[5] = regime.get("cross_tf_momentum", 0)
        features[6] = regime.get("volatility_regime", 1)
        features[7] = regime.get("trend_strength", 0)
        features[8] = regime.get("price_position", 0.5)
        features[9] = market_data.get("change_24h", 0) / 100  # normalize
        
        # Pattern features (10-14)
        features[10] = patterns.get("pv_divergence", 0)
        features[11] = patterns.get("money_flow_trend", 0)
        features[12] = patterns.get("fractal_dimension", 1.5) - 1.5  # center around 0
        features[13] = market_data.get("volume", 0) / 1_000_000  # normalize volume
        features[14] = min(1.0, market_data.get("quote_volume", 0) / 100_000_000)  # normalize
        
        # Technical features (15-19) - simplified versions of existing indicators
        if "1h" in multi_tf_data and "closes" in multi_tf_data["1h"]:
            closes = np.array(multi_tf_data["1h"]["closes"])
            if len(closes) >= 20:
                sma_20 = np.mean(closes[-20:])
                current = market_data.get("price", sma_20)
                features[15] = (current - sma_20) / sma_20  # price vs SMA
                features[16] = np.std(closes[-20:]) / sma_20  # volatility
                features[17] = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0  # short momentum
                features[18] = (np.mean(closes[-5:]) - np.mean(closes[-20:])) / np.mean(closes[-20:])  # momentum
                features[19] = (np.max(closes[-20:]) - current) / (np.max(closes[-20:]) - np.min(closes[-20:]) + 1e-6)  # position in range
        
        return np.clip(features, -5, 5)  # Prevent extreme values
        
    def detect_regime(self, features):
        """Detect market regime using feature analysis"""
        momentum = features[5]  # cross_tf_momentum
        volatility = features[6]  # volatility_regime
        trend = features[7]  # trend_strength
        
        if abs(trend) > 0.02 and volatility < 1.5:
            return 'trending_up' if trend > 0 else 'trending_down'
        elif volatility > 2.0:
            return 'volatile'
        else:
            return 'ranging'
            
    def predict_signal(self, features, regime):
        """Generate trading signal with confidence"""
        try:
            # Get regime-specific model
            model = self.regime_models.get(regime, self.regime_models['ranging'])
            
            # Simple linear model prediction
            signal = np.dot(features, model['weights']) + model['bias']
            
            # Convert to probability-like score
            signal_prob = 1 / (1 + np.exp(-signal))  # sigmoid
            
            # Calculate confidence based on feature alignment
            feature_strength = np.abs(features).mean()
            regime_consistency = self._calculate_regime_consistency(features, regime)
            
            confidence = min(0.95, feature_strength * regime_consistency)
            
            # Convert to directional signal
            if signal_prob > 0.6:
                direction = "LONG"
                strength = signal_prob - 0.5
            elif signal_prob < 0.4:
                direction = "SHORT" 
                strength = 0.5 - signal_prob
            else:
                direction = "NEUTRAL"
                strength = 0
                
            return {
                "direction": direction,
                "strength": strength * 2,  # scale to [0, 1]
                "confidence": confidence,
                "signal_prob": signal_prob,
                "regime": regime
            }
            
        except Exception as e:
            print(f"ML prediction error: {e}")
            return {"direction": "NEUTRAL", "strength": 0, "confidence": 0, "signal_prob": 0.5, "regime": "unknown"}
            
    def update_model(self, features, outcome, regime):
        """Online learning update"""
        try:
            if regime not in self.regime_models:
                return
                
            # Store for batch analysis
            self.feature_buffer.append(features.copy())
            self.outcome_buffer.append(outcome)
            
            # Online gradient update
            model = self.regime_models[regime]
            
            # Current prediction
            current_pred = np.dot(features, model['weights']) + model['bias']
            error = outcome - current_pred
            
            # Gradient descent update
            lr = self.learning_rate * (1.0 if abs(error) < 1.0 else 0.5)  # adaptive LR
            model['weights'] += lr * error * features
            model['bias'] += lr * error
            
            # Update feature importance (exponential moving average)
            feature_impact = np.abs(features * error)
            alpha = 0.1
            self.feature_importance = (1 - alpha) * self.feature_importance + alpha * feature_impact
            
            # Regularization to prevent overfitting
            model['weights'] *= 0.9999  # slight weight decay
            
            # Monitor model health
            self._check_model_health()
            
        except Exception as e:
            print(f"Model update error: {e}")
            
    def _calculate_regime_consistency(self, features, regime):
        """Check if features are consistent with detected regime"""
        momentum = features[5]
        volatility = features[6] 
        trend = features[7]
        
        if regime == 'trending_up':
            return min(1.0, max(0.2, (momentum + trend + 1) / 2))
        elif regime == 'trending_down':
            return min(1.0, max(0.2, (-momentum - trend + 1) / 2))
        elif regime == 'volatile':
            return min(1.0, max(0.3, volatility / 3))
        else:  # ranging
            return min(1.0, max(0.3, 1.0 - abs(momentum) - abs(trend)))
            
    def _check_model_health(self):
        """Monitor for overfitting"""
        if len(self.outcome_buffer) < 50:
            return
            
        recent_outcomes = list(self.outcome_buffer)[-50:]
        
        # Check for excessive variance in recent performance
        variance = np.var(recent_outcomes)
        if variance > 0.2:  # High variance indicates instability
            self.model_health = "UNSTABLE"
            self.learning_rate *= 0.8  # Reduce learning rate
        else:
            self.model_health = "HEALTHY"
            self.learning_rate = min(0.001, self.learning_rate * 1.05)

# ============== Pattern Recognition System ==============
class InstitutionalPatternDetector:
    def __init__(self):
        self.successful_patterns = {}
        self.pattern_threshold = 0.1  # Minimum edge required
        
    def detect_institutional_footprints(self, orderbook_data, market_data):
        """Detect signs of institutional activity"""
        patterns = {
            "iceberg_detected": False,
            "block_trades": False,
            "liquidity_hunting": False,
            "momentum_ignition": False
        }
        
        # Iceberg order detection (large orders hidden in small sizes)
        liquidity_score = orderbook_data.get("liquidity_score", 50)
        imbalance = orderbook_data.get("imbalance_ratio", 1.0)
        if liquidity_score > 80 and abs(imbalance - 1.0) < 0.1:
            patterns["iceberg_detected"] = True
            
        # ---- Block trade proxy (birim uyumu)
        price = float(market_data.get("price", 0.0) or 0.0)
        volume_24h = float(market_data.get("volume", 0.0) or 0.0)          # <-- eksikti
        quote_vol = float(market_data.get("quote_volume", 0.0) or 0.0)
        base_vol_est = (quote_vol / max(price, 1e-9)) if price > 0 else 0.0
        # "yüksek göreli hacim" + "sıkı spread" beraber ise block trades ihtimali
        if base_vol_est > 1.5 * volume_24h and orderbook_data.get("spread", 100) < 0.02:
            patterns["block_trades"] = True
                
        # ---- Liquidity hunting proxy
        price_change = abs(float(market_data.get("change_24h", 0.0) or 0.0))
        if price_change > 5 and liquidity_score < 30:
            patterns["liquidity_hunting"] = True

        # ---- Momentum ignition proxy: spread çok düşük + dengesizlik yüksek + anlamlı değişim
        if orderbook_data.get("spread", 99) < 0.02 and abs(imbalance - 1.0) > 0.5 and price_change > 2:
            patterns["momentum_ignition"] = True
            
        return patterns

        
    def calculate_pattern_strength(self, patterns, historical_success):
        """Calculate overall pattern strength based on historical performance"""
        strength = 0.0
        pattern_count = 0
        
        for pattern_name, detected in patterns.items():
            if detected:
                pattern_count += 1
                # Get historical success rate for this pattern
                success_rate = historical_success.get(pattern_name, 0.5)
                strength += success_rate * 0.25  # Each pattern can add up to 0.25
                
        # Bonus for multiple pattern confirmation
        if pattern_count > 1:
            strength *= 1.2
            
        return min(1.0, strength)
        
    def learn_pattern_outcome(self, patterns, outcome):
        """Update pattern success rates based on trade outcome"""
        for pattern_name, detected in patterns.items():
            if detected:
                if pattern_name not in self.successful_patterns:
                    self.successful_patterns[pattern_name] = {"wins": 0, "total": 0}
                    
                self.successful_patterns[pattern_name]["total"] += 1
                if outcome > 0:
                    self.successful_patterns[pattern_name]["wins"] += 1
                    
    def get_pattern_success_rates(self):
        """Get current success rates for all patterns"""
        rates = {}
        for pattern_name, stats in self.successful_patterns.items():
            if stats["total"] > 0:
                rates[pattern_name] = stats["wins"] / stats["total"]
            else:
                rates[pattern_name] = 0.5  # neutral prior
        return rates

# ============== Enhanced AI Decision System ==============
def ai_trade_decision_ml(coin, market_data, multi_tf_data, learning_insights):
    """
    ML-enhanced trade decision with institutional-level sophistication
    """
    try:
        current_price = market_data[coin]["price"]
        symbol = f"{coin}USDT"
        
        # Get orderbook for microstructure analysis
        orderbook_data = get_order_book_analysis(symbol)
        
        # Initialize ML components
        if not hasattr(ai_trade_decision_ml, 'ml_model'):
            ai_trade_decision_ml.ml_model = AdaptiveMLModel()
            ai_trade_decision_ml.pattern_detector = InstitutionalPatternDetector()
            
        ml_model = ai_trade_decision_ml.ml_model
        pattern_detector = ai_trade_decision_ml.pattern_detector
        
        # Extract ML features
        features = ml_model.extract_features_vector(market_data[coin], multi_tf_data, orderbook_data)
        
        # Detect market regime
        regime = ml_model.detect_regime(features)
        
        # Get ML signal
        ml_signal = ml_model.predict_signal(features, regime)
        
        # Detect institutional patterns
        institutional_patterns = pattern_detector.detect_institutional_footprints(orderbook_data, market_data[coin])
        pattern_success_rates = pattern_detector.get_pattern_success_rates()
        pattern_strength = pattern_detector.calculate_pattern_strength(institutional_patterns, pattern_success_rates)
        
        # Combine ML signal with pattern detection
        base_confidence = ml_signal["confidence"]
        pattern_boost = pattern_strength * 0.3  # Up to 30% boost
        combined_confidence = min(0.95, base_confidence + pattern_boost)
        
        # Direction determination
        if ml_signal["direction"] == "LONG" and combined_confidence > 0.6:
            direction = "LONG"
        elif ml_signal["direction"] == "SHORT" and combined_confidence > 0.6:
            direction = "SHORT"
        else:
            return {"direction": "SKIP", "reason": "ML confidence too low"}
            
        # Convert to existing system format
        confidence_score = max(6, min(10, int(combined_confidence * 10)))
        
        # Enhanced stop-loss calculation using ML features
        atr = 0.0
        if "1h" in multi_tf_data and all(k in multi_tf_data["1h"] for k in ("highs", "lows", "closes")):
            atr = calculate_atr(multi_tf_data["1h"]["highs"], multi_tf_data["1h"]["lows"], multi_tf_data["1h"]["closes"])
            
        # ML-informed risk management
        volatility_multiplier = max(0.8, min(2.0, features[6]))  # volatility_regime feature
        liquidity_factor = max(0.9, min(1.1, features[1]))  # liquidity_depth feature
        
        base_sl_mult = 1.5 * volatility_multiplier * liquidity_factor
        base_tp_mult = 2.5 * (1 + pattern_strength * 0.5)  # Higher TP when patterns detected
        
        # Calculate stops
        atr = atr or (current_price * 0.005)
        atr_sl = atr * base_sl_mult
        atr_tp = atr * base_tp_mult
        
        if direction == "LONG":
            sl = current_price - atr_sl
            tp = current_price + atr_tp
        else:
            sl = current_price + atr_sl
            tp = current_price - atr_tp
            
        sl_pct = abs(sl - current_price) / current_price * 100
        tp_pct = abs(tp - current_price) / current_price * 100
        
        # Ensure minimum risk/reward
        if tp_pct / max(sl_pct, 0.1) < 1.5:
            return {"direction": "SKIP", "reason": "RR ratio too low"}
            
        # Dynamic leverage based on ML confidence and patterns
        base_leverage = 15 + (confidence_score - 6) * 2
        pattern_leverage_boost = pattern_strength * 5  # Up to 5x additional leverage
        leverage = int(max(10, min(30, base_leverage + pattern_leverage_boost)))
        
        # Position sizing with ML confidence
        base_size = 0.05 + (confidence_score - 6) * 0.015  # 5-11%
        ml_size_mult = 1.0 + (combined_confidence - 0.6) * 0.5  # Up to 50% larger
        position_size = min(0.20, base_size * ml_size_mult)
        
        # Create reasoning with ML insights
        top_patterns = [k for k, v in institutional_patterns.items() if v]
        pattern_text = f"Patterns: {', '.join(top_patterns)}" if top_patterns else "No institutional patterns"
        
        reasoning = (
            f"ML Signal: {direction} (conf: {combined_confidence:.2f}) | "
            f"Regime: {regime} | Pattern strength: {pattern_strength:.2f} | "
            f"{pattern_text}"
        )
        
        # Store features for learning
        feature_hash = hashlib.md5(str(features).encode()).hexdigest()[:8]
        db_execute("""
            INSERT INTO ml_features (coin, features_json, regime, timestamp)
            VALUES (?, ?, ?, ?)
        """, (coin, json.dumps(features.tolist()), regime, datetime.now().isoformat()))
        
        return {
            "direction": direction,
            "leverage": leverage,
            "position_size": position_size,
            "confidence": confidence_score,
            "reasoning": reasoning,
            "stop_loss": float(sl),
            "take_profit": float(tp),
            "sl_percentage": float(sl_pct),
            "tp_percentage": float(tp_pct),
            "market_regime": regime,
            "atr": float(atr),
            "ml_confidence": float(combined_confidence),
            "pattern_strength": float(pattern_strength),
            "institutional_patterns": institutional_patterns,
            "features_hash": feature_hash,
            "overall_score": combined_confidence * 100  # For compatibility
        }
        
    except Exception as e:
        print(f"ML AI decision error: {e}")
        # Fallback to original system
        return ai_trade_decision_fallback(coin, market_data, multi_tf_data, learning_insights)

def ai_trade_decision_fallback(coin, market_data, multi_tf_data, learning_insights):
    """Fallback to original deterministic system if ML fails"""
    try:
        current_price = market_data[coin]["price"]
        symbol = f"{coin}USDT"
        
        # Simple scoring fallback
        score = 50.0
        if market_data[coin]["change_24h"] > 5:
            score += 20
        elif market_data[coin]["change_24h"] < -5:
            score -= 20
            
        if score >= 65:
            direction = "LONG"
        elif score <= 35:
            direction = "SHORT"
        else:
            return {"direction": "SKIP", "reason": "Fallback: neutral conditions"}
            
        confidence = max(6, min(10, int(abs(score - 50) / 5)))
        leverage = 15
        
        # Simple stops
        sl_pct = 2.0
        tp_pct = 4.0
        
        if direction == "LONG":
            sl = current_price * (1 - sl_pct / 100)
            tp = current_price * (1 + tp_pct / 100)
        else:
            sl = current_price * (1 + sl_pct / 100)
            tp = current_price * (1 - tp_pct / 100)
            
        return {
            "direction": direction,
            "leverage": leverage,
            "position_size": 0.08,
            "confidence": confidence,
            "reasoning": f"Fallback system: {direction} based on price change",
            "stop_loss": float(sl),
            "take_profit": float(tp),
            "sl_percentage": sl_pct,
            "tp_percentage": tp_pct,
            "market_regime": "unknown",
            "atr": current_price * 0.02,
            "ml_confidence": 0.5,
            "pattern_strength": 0.0,
            "overall_score": score
        }
        
    except Exception as e:
        print(f"Fallback system error: {e}")
        return {"direction": "SKIP", "reason": "All systems failed"}

# ============== ML Learning Integration ==============
def update_ml_models_on_trade_close(coin, direction, pnl_percent, features_hash):
    """Update ML models when a trade closes"""
    try:
        # Get the ML model instance
        if hasattr(ai_trade_decision_ml, 'ml_model') and hasattr(ai_trade_decision_ml, 'pattern_detector'):
            ml_model = ai_trade_decision_ml.ml_model
            pattern_detector = ai_trade_decision_ml.pattern_detector
            
            # En son feature set'ini al (hash'a bağlı kalma)
            rows = db_execute("""
                SELECT features_json, regime FROM ml_features 
                WHERE coin=? 
                ORDER BY timestamp DESC LIMIT 1
            """, (coin,), fetch=True)
            
            if rows:
                features_json, regime = rows[0]
                features = np.array(json.loads(features_json))

                # Normalize outcome to [-1, 1]
                normalized_outcome = np.tanh(pnl_percent / 10.0)

                # Update the model
                ml_model.update_model(features, normalized_outcome, regime)
                print(f"📚 Updated ML model: {coin} {direction} -> {pnl_percent:.1f}% (regime: {regime})")

        # Pattern öğrenmesi (mevcut mantığı koruyoruz)
        rows = db_execute("""
            SELECT reasoning FROM ai_decisions 
            WHERE coin=? AND direction=? AND outcome_pnl IS NULL
            ORDER BY timestamp DESC LIMIT 1
        """, (coin, direction), fetch=True)
        
        if rows and "Patterns:" in rows[0][0]:
            if hasattr(ai_trade_decision_ml, 'pattern_detector'):
                pattern_detector = ai_trade_decision_ml.pattern_detector
                dummy_patterns = {"momentum_ignition": True}
                pattern_detector.learn_pattern_outcome(dummy_patterns, pnl_percent)

    except Exception as e:
        print(f"ML model update error: {e}")


# ============== Enhanced Position Monitoring ==============
def calculate_ml_exit_signal(position_data, multi_tf_data):
    """Use ML to determine optimal exit timing"""
    try:
        if not hasattr(ai_trade_decision_ml, 'ml_model'):
            return False, "ML not initialized"
            
        coin = position_data["coin"]
        symbol = position_data["symbol"]
        price = position_data["mark_price"]
        
        # Create market data structure
        market_data = {coin: {"price": price, "change_24h": 0, "volume": 0}}
        orderbook_data = get_order_book_analysis(symbol)
        
        ml_model = ai_trade_decision_ml.ml_model
        
        # Extract current features
        features = ml_model.extract_features_vector(market_data[coin], multi_tf_data, orderbook_data)
        regime = ml_model.detect_regime(features)
        
        # Get ML signal for current conditions
        ml_signal = ml_model.predict_signal(features, regime)
        
        direction = position_data["direction"]
        entry_price = position_data["entry_price"]
        
        # Calculate current PnL
        if direction == "LONG":
            pnl_pct = (price - entry_price) / entry_price
            # Exit if ML suggests SHORT and we have decent profit
            should_exit = (ml_signal["direction"] == "SHORT" and 
                          ml_signal["confidence"] > 0.7 and 
                          pnl_pct > 0.02)
        else:
            pnl_pct = (entry_price - price) / entry_price  
            # Exit if ML suggests LONG and we have decent profit
            should_exit = (ml_signal["direction"] == "LONG" and 
                          ml_signal["confidence"] > 0.7 and 
                          pnl_pct > 0.02)
        
        if should_exit:
            reason = f"ML reversal signal: {ml_signal['direction']} conf={ml_signal['confidence']:.2f} regime={regime}"
            return True, reason
            
        # Also exit if regime changed unfavorably
        if regime == 'volatile' and ml_signal["confidence"] > 0.8 and pnl_pct > 0.01:
            return True, f"Regime turned volatile (conf={ml_signal['confidence']:.2f})"
            
        return False, None
        
    except Exception as e:
        print(f"ML exit signal error: {e}")
        return False, None

# ============== All Original Functions (Updated) ==============
def get_futures_balance():
    try:
        info = binance_request("/fapi/v2/account", "GET", signed=True)
        if not info: return 0.0
        total = float(info.get("totalWalletBalance", 0))
        print(f"💰 Futures Total Wallet: {total:.2f} USDT")
        return total
    except Exception as e:
        print(f"Balance error: {e}")
        return 0.0

def get_open_positions():
    try:
        positions = binance_request("/fapi/v2/positionRisk", "GET", signed=True) or []
        out = {}
        for p in positions:
            amt = float(p.get("positionAmt", 0))
            if abs(amt) <= 0: continue
            sym = p["symbol"]
            coin = sym.replace("USDT","").replace("USDC","").replace("BNFCR","")
            entry = float(p.get("entryPrice") or p.get("avgPrice") or 0)
            mark  = float(p.get("markPrice") or p.get("lastPrice") or 0)
            pnl   = float(p.get("unRealizedPnl") or p.get("unrealizedPnl") or 0)
            lev   = int(float(p.get("leverage", "1")))
            out[sym] = {
                "coin": coin, "symbol": sym, "size": amt,
                "direction": "LONG" if amt>0 else "SHORT",
                "entry_price": entry, "mark_price": mark, "pnl": pnl,
                "leverage": lev, "notional": abs(amt*mark)
            }
        return out
    except Exception as e:
        print(f"Positions error: {e}")
        return {}

def get_batch_market_data():
    try:
        data = binance_request("/fapi/v1/ticker/24hr", "GET", signed=False) or []
        target = {'BTC','ETH','SOL','BNB','SEI','DOGE','TIA','TAO','ARB','SUI','ENA','FET'}
        md = {}
        for t in data:
            sym = t["symbol"]
            if not sym.endswith("USDT"): continue
            coin = sym[:-4]
            if coin in target:
                md[coin] = {
                    "price": float(t["lastPrice"]),
                    "change_24h": float(t["priceChangePercent"]),
                    "volume": float(t["volume"]),
                    "high_24h": float(t["highPrice"]),
                    "low_24h": float(t["lowPrice"]),
                    "quote_volume": float(t["quoteVolume"])
                }
        return md
    except Exception as e:
        print(f"24hr fetch error: {e}")
        return {}

def get_multi_timeframe_data(symbol):
    intervals = {'5m':('5m',48),'15m':('15m',48),'1h':('1h',48),'4h':('4h',42),'1d':('1d',30)}
    out = {}
    for k,(itv,limit) in intervals.items():
        try:
            candles = binance_request("/fapi/v1/klines","GET",params={"symbol":symbol,"interval":itv,"limit":limit}) or []
            out[k] = {
                "candles": candles,
                "closes": [float(c[4]) for c in candles],
                "volumes":[float(c[5]) for c in candles],
                "highs":  [float(c[2]) for c in candles],
                "lows":   [float(c[3]) for c in candles],
                "opens":  [float(c[1]) for c in candles]
            }
        except:
            pass
    return out

def get_order_book_analysis(symbol):
    try:
        data = binance_request("/fapi/v1/depth", "GET", params={"symbol": symbol, "limit": 100})
        raw_bids = data.get("bids", [])[:20]
        raw_asks = data.get("asks", [])[:20]

        def _pairs(raw):
            out = []
            for row in raw:
                if len(row) >= 2:
                    out.append([float(row[0]), float(row[1])])
            return out

        bids = _pairs(raw_bids)
        asks = _pairs(raw_asks)
        if not bids or not asks:
            return {"liquidity_score": 50, "imbalance_ratio": 1.0, "spread": 0}

        best_bid = bids[0][0]
        best_ask = asks[0][0]
        spread_pct = ((best_ask - best_bid) / best_bid) * 100 if best_bid else 0.0

        bid_notional = sum(p * q for p, q in bids[:10])
        ask_notional = sum(p * q for p, q in asks[:10])
        total_notional = bid_notional + ask_notional
        imbalance = (bid_notional / ask_notional) if ask_notional > 0 else 1.0

        liq_score = min(100.0, (total_notional / 1_000_000.0) * 100.0)

        return {
            "liquidity_score": liq_score,
            "imbalance_ratio": imbalance,
            "spread": spread_pct,
            "bid_notional": bid_notional,
            "ask_notional": ask_notional
        }
    except Exception as e:
        print(f"Orderbook error: {e}")
        return {"liquidity_score": 50, "imbalance_ratio": 1.0, "spread": 0}

def calculate_atr(highs, lows, closes, period=14):
    if len(highs) < period+1 or len(lows) < period+1 or len(closes) < period+1: return 0
    tr = []
    for i in range(1,len(highs)):
        hl = highs[i]-lows[i]
        hc = abs(highs[i]-closes[i-1])
        lc = abs(lows[i]-closes[i-1])
        tr.append(max(hl,hc,lc))
    if len(tr)<period: return 0
    return sum(tr[-period:]) / period

# ============== Position Management ==============
def place_futures_order(symbol, side, quantity, leverage=10, order_type="MARKET"):
    try:
        _ = binance_request("/fapi/v1/leverage","POST",params={"symbol":symbol,"leverage":leverage},signed=True)
        cid = f"ml-bot-{int(time.time()*1000)}-{side}"
        params = {"symbol":symbol,"side":side,"type":order_type,"quantity":quantity,"newClientOrderId":cid}
        res = binance_request("/fapi/v1/order","POST",params=params,signed=True)
        return res
    except Exception as e:
        print(f"Place order error: {e}")
        return None

def close_futures_position(symbol):
    try:
        positions = get_open_positions()
        if symbol not in positions: 
            print(f"No open position for {symbol}")
            return None
        pos = positions[symbol]; size = abs(pos["size"])
        side = "SELL" if pos["direction"]=="LONG" else "BUY"
        cid = f"ml-close-{int(time.time()*1000)}"
        res = binance_request("/fapi/v1/order","POST",signed=True,params={
            "symbol":symbol,"side":side,"type":"MARKET","quantity":q_qty(symbol,size),
            "reduceOnly":True,"newClientOrderId":cid
        })
        return res
    except Exception as e:
        print(f"Close error: {e}")
        return None

def set_stop_loss_take_profit(symbol, stop_price, take_profit_price, position_side):
    try:
        positions = get_open_positions()
        if symbol not in positions:
            return None
        qty = abs(positions[symbol]["size"])
        qty = q_qty(symbol, qty)

        sl_side = "SELL" if position_side == "LONG" else "BUY"
        tp_side = sl_side

        sl_order = binance_request("/fapi/v1/order","POST",signed=True,params={
            "symbol": symbol, "side": sl_side, "type": "STOP_MARKET", "quantity": qty,
            "stopPrice": q_price(symbol, stop_price), "workingType": "MARK_PRICE", "reduceOnly": True
        })
        tp_order = binance_request("/fapi/v1/order","POST",signed=True,params={
            "symbol": symbol, "side": tp_side, "type": "TAKE_PROFIT_MARKET", "quantity": qty,
            "stopPrice": q_price(symbol, take_profit_price), "workingType": "MARK_PRICE", "reduceOnly": True
        })
        return {"stop_loss": sl_order, "take_profit": tp_order}
    except Exception as e:
        print(f"SL/TP error: {e}")
        return None

# ============== Learning & Stats ==============
def save_ai_decision(coin, trade_params):
    db_execute("""
        INSERT INTO ai_decisions (coin,direction,confidence,reasoning,market_regime,ml_confidence,pattern_strength,regime_detected,features_hash,timestamp)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (coin, trade_params["direction"], trade_params["confidence"], 
          (trade_params.get("reasoning") or "")[:200], trade_params.get("market_regime","UNKNOWN"),
          trade_params.get("ml_confidence", 0.5), trade_params.get("pattern_strength", 0.0),
          trade_params.get("market_regime", "unknown"), trade_params.get("features_hash", ""),
          datetime.now().isoformat()))

def update_ai_decision_outcome(coin, direction, pnl_percent):
    db_execute("""
        UPDATE ai_decisions SET outcome_pnl=?
        WHERE coin=? AND direction=? AND outcome_pnl IS NULL
        ORDER BY timestamp DESC LIMIT 1
    """, (pnl_percent, coin, direction))
    
    # Update ML models with outcome
    rows = db_execute("""
        SELECT features_hash FROM ai_decisions 
        WHERE coin=? AND direction=? AND outcome_pnl=?
        ORDER BY timestamp DESC LIMIT 1
    """, (coin, direction, pnl_percent), fetch=True)
    
    if rows:
        features_hash = rows[0][0]
        update_ml_models_on_trade_close(coin, direction, pnl_percent, features_hash)

def update_coin_stats(coin, pnl_percent):
    alpha = 0.15
    rows = db_execute("SELECT trades, wins, sum_pnl, ema_wr, ema_pnl FROM coin_stats WHERE coin=?", (coin,), fetch=True)
    if not rows:
        db_execute("INSERT INTO coin_stats (coin, trades, wins, sum_pnl, ema_wr, ema_pnl) VALUES (?,?,?,?,?,?)",
                   (coin, 0, 0, 0.0, 0.5, 0.0))
        trades = wins = 0; sum_pnl = 0.0; ema_wr = 0.5; ema_pnl = 0.0
    else:
        trades, wins, sum_pnl, ema_wr, ema_pnl = rows[0]

    trades += 1
    wins += 1 if pnl_percent > 0 else 0
    sum_pnl += float(pnl_percent)
    wr = wins / max(1, trades)
    ema_wr = (1 - alpha) * float(ema_wr) + alpha * wr
    ema_pnl = (1 - alpha) * float(ema_pnl) + alpha * float(pnl_percent)

    db_execute("UPDATE coin_stats SET trades=?, wins=?, sum_pnl=?, ema_wr=?, ema_pnl=? WHERE coin=?",
               (trades, wins, sum_pnl, ema_wr, ema_pnl, coin))

def rank_candidates_ucb(coin_scores, c=2.0):
    trow = db_execute("SELECT COALESCE(SUM(trades),0) FROM coin_stats", fetch=True)
    total_trades = float(trow[0][0] if trow and trow[0] else 0.0) + 1.0

    out = []
    for coin, score, ana, tf in coin_scores:
        row = db_execute("SELECT trades, ema_pnl FROM coin_stats WHERE coin=?", (coin,), fetch=True)
        if row:
            n_i, ema_p = float(row[0][0] or 0.0), float(row[0][1] or 0.0)
        else:
            n_i, ema_p = 0.0, 0.0
        explore = c * math.sqrt(math.log(total_trades) / max(1.0, n_i))
        ucb = ema_p + explore
        out.append((ucb, coin, score, ana, tf))
    out.sort(key=lambda x: (x[0], abs(x[2] - 50)), reverse=True)
    return [(coin, score, ana, tf) for (_ucb, coin, score, ana, tf) in out]

def get_learning_insights():
    insights={}
    total = db_execute("SELECT COUNT(*) FROM trades WHERE pnl IS NOT NULL", fetch=True)[0][0]
    wins  = db_execute("SELECT COUNT(*) FROM trades WHERE pnl > 0", fetch=True)[0][0]
    insights["win_rate"] = (wins/total*100) if total>0 else 0
    insights["total_trades"] = total
    row = db_execute("""
        SELECT leverage, AVG(pnl_percent) FROM trades 
        WHERE pnl IS NOT NULL GROUP BY leverage ORDER BY AVG(pnl_percent) DESC LIMIT 1
    """, fetch=True)
    insights["best_leverage"] = row[0][0] if row else 15
    avgp = db_execute("SELECT AVG(pnl) FROM trades WHERE pnl > 0", fetch=True)[0][0]
    avgl = db_execute("SELECT AVG(pnl) FROM trades WHERE pnl < 0", fetch=True)[0][0]
    insights["avg_profit"] = avgp or 0
    insights["avg_loss"] = avgl or 0
    return insights

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def get_policy_adjustments():
    rows = db_execute("""
        SELECT coin, outcome_pnl FROM ai_decisions
        WHERE outcome_pnl IS NOT NULL
        ORDER BY timestamp DESC LIMIT 500
    """, fetch=True) or []

    total = len(rows)
    wins = sum(1 for _, pnl in rows if (pnl or 0) > 0)
    global_wr = (wins / total) if total else 0.5

    shift = clamp((global_wr - 0.50) * 10.0, -3.0, 2.0)
    bullish_thr = 65.0 - shift
    bearish_thr = 35.0 + shift

    per = {}
    for coin, pnl in rows:
        if coin not in per:
            per[coin] = {"n": 0, "wins": 0, "sum_pnl": 0.0}
        per[coin]["n"] += 1
        per[coin]["wins"] += 1 if (pnl or 0) > 0 else 0
        per[coin]["sum_pnl"] += float(pnl or 0.0)

    per_coin = {}
    for coin, st in per.items():
        n = st["n"]
        wr = st["wins"] / n if n else 0.5
        avg_pnl = (st["sum_pnl"] / n) if n else 0.0

        bias = clamp((wr - 0.5) * 20.0, -3.0, 3.0)
        size_mult = clamp(1.0 + (wr - 0.5) * 0.30, 0.85, 1.15)
        lev_mult  = clamp(1.0 + (avg_pnl / 100.0) * 0.50, 0.90, 1.20)

        per_coin[coin] = {"bias": bias, "size_mult": size_mult, "lev_mult": lev_mult}

    row = db_execute("SELECT value FROM hyperparameters WHERE key='exploration_rate'", fetch=True)
    exploration_rate = float(row[0][0]) if row and row[0] and row[0][0] is not None else 0.10
    exploration_rate = clamp(exploration_rate, 0.00, 0.30)

    return {
        "global": {"win_rate": global_wr, "bullish_thr": bullish_thr, "bearish_thr": bearish_thr, "exploration_rate": exploration_rate},
        "per_coin": per_coin
    }

def get_hp_value(key, default):
    row = db_execute("SELECT value FROM hyperparameters WHERE key=?", (key,), fetch=True)
    try:
        return float(row[0][0]) if row and row[0] and row[0][0] is not None else default
    except:
        return default

def get_realized_pnl_today():
    rows = db_execute("""
        SELECT COALESCE(SUM(pnl), 0) FROM trades
        WHERE action LIKE '%CLOSE%' AND DATE(timestamp) = DATE('now')
    """, fetch=True)
    return float(rows[0][0] if rows and rows[0] and rows[0][0] is not None else 0.0)

def calculate_portfolio_risk():
    try:
        positions = get_open_positions()
        balance = get_futures_balance()
        if balance <= 0:
            return {"can_trade":False,"reason":"No balance available","positions_count":0,"balance":0,
                    "margin_ratio":0,"available_balance":0, "can_trade_long":False,"can_trade_short":False}

        total_notional = sum(p["notional"] for p in positions.values())
        total_margin   = sum(p["notional"]/p["leverage"] for p in positions.values())
        margin_ratio = total_margin / balance if balance>0 else 0
        available = balance - total_margin

        long_notional  = sum(p["notional"] for p in positions.values() if p["direction"]=="LONG")
        short_notional = sum(p["notional"] for p in positions.values() if p["direction"]=="SHORT")
        max_dir_x = get_hp_value("max_directional_exposure_x", 6.0)
        can_long  = (long_notional / max(1e-9, balance))  < max_dir_x
        can_short = (short_notional / max(1e-9, balance)) < max_dir_x

        max_dd_pct = get_hp_value("max_daily_loss_pct", 8.0)
        realized_today = get_realized_pnl_today()
        dd_hit = (realized_today <= -abs(max_dd_pct) * balance / 100.0)

        out = {"positions_count":len(positions), "total_notional":total_notional,"total_margin_used":total_margin,
               "balance":balance,"margin_ratio":margin_ratio,"available_balance":available,
               "max_positions":MAX_CONCURRENT_POSITIONS, "can_trade":True,"reason":"Good to trade",
               "can_trade_long":can_long and not dd_hit, "can_trade_short":can_short and not dd_hit,
               "realized_today":realized_today}

        if len(positions) >= MAX_CONCURRENT_POSITIONS:
            out["can_trade"]=False; out["reason"]=f"Maximum positions reached ({len(positions)}/{MAX_CONCURRENT_POSITIONS})"
        elif margin_ratio > 0.9:
            out["can_trade"]=False; out["reason"]=f"High margin utilization ({margin_ratio*100:.1f}%)"
        elif available < 5:
            out["can_trade"]=False; out["reason"]=f"Insufficient available balance (${available:.2f})"
        elif dd_hit:
            out["can_trade"]=False; out["reason"]=f"Daily loss breaker hit (PnL today {realized_today:.2f})"

        return out
    except Exception as e:
        print(f"Risk error: {e}")
        return {"can_trade":False,"reason":f"Risk calculation error: {e}","can_trade_long":False,"can_trade_short":False}

def get_dynamic_position_size(current_positions_count, confidence_score, balance):
    try:
        base = POSITION_SCALING.get(current_positions_count, 0.04)
        conf_mult = 0.8 + (confidence_score - 6)*0.09
        positions = get_open_positions()
        used_margin = sum(p["notional"]/p["leverage"] for p in positions.values())
        available = balance - used_margin
        max_val = available * 0.99
        target_val = balance * base * conf_mult
        final_size = min(target_val, max_val) / balance if balance>0 else 0.07
        final_size = max(0.04, min(0.22, final_size))
        return {"position_size":final_size,"base_size":base,"confidence_multiplier":conf_mult,
                "available_balance":available,"position_value":balance*final_size}
    except Exception as e:
        print(f"Dynamic size error: {e}")
        return {"position_size":0.05,"position_value":balance*0.05 if balance else 0}

# ============== Enhanced Trade Execution ==============
def execute_ml_trade(coin, trade_params, current_price, bullish_thr=65.0, bearish_thr=35.0, 
                     size_mult=1.0, lev_mult=1.0, score_bias=0.0):
    """Execute trade with ML enhancements and aggressive positioning"""
    if trade_params["direction"] == "SKIP":
        return False
        
    try:
        symbol = f"{coin}USDT"
        
        # Use ML confidence for enhanced gating
        ml_confidence = trade_params.get("ml_confidence", 0.5)
        pattern_strength = trade_params.get("pattern_strength", 0.0)
        
        # Dynamic threshold adjustment based on ML signals
        if ml_confidence > 0.8:  # Very confident ML signal
            bullish_thr -= 5  # More aggressive entry
            bearish_thr += 5
        elif ml_confidence < 0.6:  # Low ML confidence
            bullish_thr += 3  # More conservative
            bearish_thr -= 3
            
        base_score = trade_params.get("overall_score", 50.0)
        score = base_score + float(score_bias or 0.0)

        # ML-enhanced gating logic
        if trade_params["direction"] == "LONG":
            ml_boost = ml_confidence * pattern_strength * 10  # Up to 10 points boost
            effective_score = score + ml_boost
            if effective_score < bullish_thr:
                print(f"Reject LONG: score {effective_score:.1f} < gate {bullish_thr:.1f} (ML boost: {ml_boost:.1f})")
                return False

        elif trade_params["direction"] == "SHORT":
            ml_boost = ml_confidence * pattern_strength * 10
            # SHORT'ta long-odaklı skoru ters çevirerek değerlendiriyoruz
            short_score = (100.0 - score) + ml_boost
            if short_score < bearish_thr:
                print(f"Reject SHORT: short_score {short_score:.1f} < gate {bearish_thr:.1f} (ML boost: +{ml_boost:.1f})")
                return False

        # Enhanced risk management
        prisk = calculate_portfolio_risk()
        if not prisk["can_trade"]:
            print(f"Cannot trade: {prisk['reason']}")
            return False
            
        if trade_params["direction"] == "LONG" and not prisk.get("can_trade_long", True):
            print("Directional cap: cannot open more LONG exposure")
            return False
        if trade_params["direction"] == "SHORT" and not prisk.get("can_trade_short", True):
            print("Directional cap: cannot open more SHORT exposure")
            return False

        balance = prisk["balance"]
        current_positions = get_open_positions()

        # ML-enhanced position sizing
        base_sizing = get_dynamic_position_size(len(current_positions), trade_params["confidence"], balance)
        
        # Pattern strength multiplier (aggressive)
        pattern_mult = 1.0 + (pattern_strength * 0.4)  # Up to 40% larger for strong patterns
        
        # ML confidence multiplier (very aggressive)
        ml_mult = 1.0 + ((ml_confidence - 0.6) * 0.6)  # Up to 60% larger for high confidence
        
        # Combined multipliers
        total_mult = float(size_mult or 1.0) * pattern_mult * ml_mult
        position_value = base_sizing["position_value"] * total_mult
        
        # Cap at aggressive but safe limits
        max_position_pct = 0.25  # 25% of account max
        position_value = min(position_value, balance * max_position_pct)

        # ML-enhanced leverage
        base_leverage = trade_params.get("leverage", 15)
        
        # Pattern-based leverage boost
        pattern_lev_boost = pattern_strength * 8  # Up to 8x additional
        
        # ML confidence leverage boost  
        ml_lev_boost = (ml_confidence - 0.6) * 10  # Up to 10x additional for high confidence
        
        # Regime-based leverage
        regime = trade_params.get("market_regime", "unknown")
        regime_mult = {"trending_up": 1.2, "trending_down": 1.2, "ranging": 0.9, "volatile": 0.8}.get(regime, 1.0)
        
        final_leverage = int(base_leverage + pattern_lev_boost + ml_lev_boost)
        final_leverage = int(final_leverage * regime_mult * float(lev_mult or 1.0))
        final_leverage = max(10, min(50, final_leverage))  # Aggressive range: 10-50x

        # Position sizing
        notional = position_value * final_leverage
        qty = q_qty(symbol, notional / current_price)
        
        if qty <= 0:
            print("Qty rounded to zero")
            return False

        # Minimum notional check
        min_notional = _symbol_info.get(symbol, {}).get("minNotional", Decimal("5"))
        computed_notional = Decimal(str(current_price)) * Decimal(str(qty))
        if computed_notional < min_notional:
            print(f"Below MIN_NOTIONAL: notional={float(computed_notional):.4f} < {float(min_notional):.4f}")
            return False

        print(f"🤖 ML TRADE #{len(current_positions)+1}: {trade_params['direction']} {coin}")
        print(f"   💰 Size: {position_value:.0f} USDT ({position_value/balance*100:.1f}% of account)")
        print(f"   📊 Leverage: {final_leverage}x (base: {base_leverage}x)")
        print(f"   🧠 ML Confidence: {ml_confidence:.2f} | Pattern Strength: {pattern_strength:.2f}")
        print(f"   📈 Expected Notional: ${float(computed_notional):.0f}")
        print(f"   🎯 R/R: 1:{trade_params.get('tp_percentage', 0)/max(trade_params.get('sl_percentage', 1), 0.1):.1f}")

        # Execute the trade
        side = "BUY" if trade_params["direction"] == "LONG" else "SELL"
        res = place_futures_order(symbol, side, qty, final_leverage)
        
        if not res:
            print(f"Order failed for {symbol}")
            return False

        # Enhanced SL/TP with ML adjustments
        sl_price = trade_params["stop_loss"]
        tp_price = trade_params["take_profit"]
        
        # Adjust stops based on ML confidence
        if ml_confidence > 0.8:
            # Tighter stops when very confident
            sl_distance = abs(sl_price - current_price)
            tp_distance = abs(tp_price - current_price)
            
            sl_price = current_price + (sl_distance * 0.9 * (1 if trade_params["direction"] == "SHORT" else -1))
            tp_price = current_price + (tp_distance * 1.1 * (1 if trade_params["direction"] == "LONG" else -1))

        sltp = set_stop_loss_take_profit(symbol, sl_price, tp_price, trade_params["direction"])
        if not sltp or not sltp.get("stop_loss") or not sltp.get("take_profit"):
            print("SL/TP placement failed, retrying once...")
            sltp = set_stop_loss_take_profit(symbol, sl_price, tp_price, trade_params["direction"])
            if not sltp:
                print("SL/TP failed twice — closing immediately")
                close_futures_position(symbol)
                return False

        # Enhanced trade logging
        save_ai_decision(coin, trade_params)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pid = f"{symbol}_{ts.replace(' ','_').replace(':','-')}"
        
        db_execute("""
            INSERT OR REPLACE INTO trades (timestamp,position_id,coin,action,direction,price,position_size,leverage,
            notional_value,stop_loss,take_profit,confidence,reason,market_conditions)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (ts, pid, coin, f"{trade_params['direction']} OPEN", trade_params["direction"], current_price,
              position_value, final_leverage, float(computed_notional), sl_price, tp_price, 
              trade_params["confidence"], trade_params.get("reasoning",""), 
              f"ML:{ml_confidence:.2f} Pattern:{pattern_strength:.2f} Regime:{regime}"))

        print(f"✅ EXECUTED: {coin} {trade_params['direction']} | Aggressive ML positioning")
        return True

    except Exception as e:
        print(f"ML Trade execution error: {e}")
        return False

# ============== Enhanced Position Monitoring ==============
def monitor_ml_positions():
    """Enhanced position monitoring with ML exit signals"""
    try:
        positions = get_open_positions()
        
        for symbol, pos in positions.items():
            mtd = get_multi_timeframe_data(symbol)
            
            # Traditional exit check
            should_close_traditional, traditional_reason = check_profit_taking_opportunity_enhanced(pos, mtd)
            
            # ML-enhanced exit check
            should_close_ml, ml_reason = calculate_ml_exit_signal(pos, mtd)
            
            # Combine exit signals
            should_close = should_close_traditional or should_close_ml
            reason = ml_reason if should_close_ml else traditional_reason
            
            if should_close:
                print(f"🧠 ML Exit {pos['coin']}: {reason}")
                res = close_futures_position(symbol)
                
                if res:
                    print(f"✅ Closed: {pos['coin']}")
                    
                    # Calculate PnL
                    realized_usd = float(pos["pnl"])
                    pnl_percent = (pos["pnl"] / (pos["notional"]/max(1,pos["leverage"]))) * 100 if pos["notional"]>0 else 0.0
                    
                    # Update learning systems
                    update_ai_decision_outcome(pos["coin"], pos["direction"], pnl_percent)
                    update_coin_stats(pos["coin"], pnl_percent)

                    # Enhanced trade logging
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    db_execute("""
                        INSERT INTO trades (timestamp, position_id, coin, action, direction, price,
                                            position_size, leverage, notional_value, stop_loss, take_profit,
                                            pnl, pnl_percent, reason, confidence, profitable, market_conditions)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (ts, f"{symbol}_CLOSE_{ts.replace(' ','_').replace(':','-')}",
                          pos["coin"], f"{pos['direction']} CLOSE", pos["direction"], pos["mark_price"],
                          None, pos["leverage"], pos["notional"], None, None,
                          realized_usd, pnl_percent, reason, None, 1 if pnl_percent>0 else 0,
                          "ML_Enhanced_Exit"))
                          
                else:
                    print(f"❌ Close failed: {pos['coin']}")
                    
        return positions
        
    except Exception as e:
        print(f"ML Monitor error: {e}")
        return {}

def check_profit_taking_opportunity_enhanced(position_data, multi_tf_data):
    """Enhanced profit taking with more aggressive thresholds"""
    price = position_data["mark_price"]
    entry = position_data["entry_price"] 
    direction = position_data["direction"]
    pnl_pct = (price-entry)/entry if direction=="LONG" else (entry-price)/entry
    
    if pnl_pct <= 0.01:  # More aggressive minimum (1% vs 1.5%)
        return (False, None)

    # More aggressive profit taking
    if pnl_pct > 0.12:  # Take profits at 12% (vs 8%)
        return (True, f"Aggressive profit taking: {pnl_pct*100:.1f}%")
        
    # Quick scalp opportunities
    if pnl_pct > 0.03 and position_data.get("leverage", 15) > 25:
        return (True, f"High leverage scalp: {pnl_pct*100:.1f}% at {position_data.get('leverage')}x")
        
    return (False, None)

# ============== Cost Tracking ==============
cost_tracker = {'ml_tokens':0,'ml_cost':0.0,'api_calls_count':0,'railway_start_time':datetime.now(),'daily_costs':[],'last_reset':datetime.now()}

def get_cost_projections():
    rows = db_execute("""
        SELECT AVG(openai_cost), AVG(api_calls) FROM daily_cost_summary
        WHERE date >= date('now','-7 days')
    """, fetch=True)
    avg_cost = rows[0][0] if rows and rows[0] else None
    hours = (datetime.now()-cost_tracker["railway_start_time"]).total_seconds()/3600
    railway_hourly = 0.01
    current_railway = hours*railway_hourly
    
    if avg_cost is not None:
        daily_cost = avg_cost
        weekly_cost = daily_cost*7
        monthly_cost = daily_cost*30
    else:
        if hours>0:
            daily_cost = (cost_tracker["ml_cost"]/hours)*24
            weekly_cost = daily_cost*7
            monthly_cost = daily_cost*30
        else:
            daily_cost=weekly_cost=monthly_cost=0
            
    weekly_rail = railway_hourly*24*7
    monthly_rail = railway_hourly*24*30
    
    return {
        "current":{"ml":cost_tracker["ml_cost"],"railway":current_railway,
                   "total":cost_tracker["ml_cost"]+current_railway,"api_calls":cost_tracker["api_calls_count"]},
        "projections":{"weekly":{"ml":weekly_cost,"railway":weekly_rail,"total":weekly_cost+weekly_rail},
                       "monthly":{"ml":monthly_cost,"railway":monthly_rail,"total":monthly_cost+monthly_rail}}
    }

def calculate_portfolio_value(market_data):
    balance = get_futures_balance()
    positions = get_open_positions()
    total_pnl = sum(p["pnl"] for p in positions.values())
    return balance + total_pnl

# ============== Flask API ==============
app = Flask(__name__)

@app.route("/")
def dashboard():
    return send_from_directory(".", "index.html")

@app.route("/dashboard.js") 
def dashboard_js():
    return send_from_directory(".", "dashboard.js")

@app.route("/api/status")
def api_status():
    try:
        portfolio["balance"] = get_futures_balance()
        portfolio["positions"] = get_open_positions()
        md = get_batch_market_data()
        total_value = calculate_portfolio_value(md)
        insights = get_learning_insights()
        cost_proj = get_cost_projections()
        
        # ML-specific metrics
        ml_metrics = {}
        if hasattr(ai_trade_decision_ml, 'ml_model'):
            ml_model = ai_trade_decision_ml.ml_model
            ml_metrics = {
                "model_health": ml_model.model_health,
                "learning_rate": ml_model.learning_rate,
                "feature_buffer_size": len(ml_model.feature_buffer),
                "total_regime_models": len(ml_model.regime_models)
            }
        
        return jsonify({
            "total_value": total_value,
            "balance": portfolio["balance"], 
            "positions": portfolio["positions"],
            "trade_history": portfolio["trade_history"][-20:],
            "market_data": md,
            "learning_metrics": insights,
            "cost_tracking": cost_proj,
            "ml_metrics": ml_metrics,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def run_flask_app():
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)

# ============== Main ML-Enhanced Trading Loop ==============
def run_ml_enhanced_bot():
    """Main trading loop with institutional-level ML"""
    init_database()
    sync_server_time() 
    load_exchange_info()

    print("🤖 INSTITUTIONAL ML TRADING BOT - AGGRESSIVE MODE")
    print("   🧠 Features: Online Learning, Pattern Recognition, Regime Detection")
    print("   ⚡ Mode: Maximum Aggression with Smart Risk Management")
    
    balance = get_futures_balance()
    if balance <= 0:
        print("❌ No futures balance or API error")
        return
        
    print(f"✅ Connected | Balance: ${balance:.2f}")
    print(f"🎯 Target: Institutional-grade performance with 15-50x leverage")

    target_coins = ['BTC','ETH','SOL','BNB','SEI','DOGE','TIA','TAO','ARB','SUI','ENA','FET']
    last_full = 0
    iteration = 0

    while True:
        try:
            iteration += 1
            now = time.time()
            
            # Enhanced position monitoring
            current_positions = monitor_ml_positions()

            if now - last_full >= 180:  # Every 3 minutes
                print("\n" + "="*90)
                print(f"🤖 ML ANALYSIS #{iteration} @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("="*90)

                current_balance = get_futures_balance()
                portfolio["balance"] = current_balance
                portfolio["positions"] = current_positions
                md = get_batch_market_data()

                print("\n💰 Account Status:")
                print(f"   Balance: ${current_balance:.2f}")
                print(f"   Open Positions: {len(current_positions)}")
                if current_positions:
                    total_pnl = sum(p['pnl'] for p in current_positions.values())
                    print(f"   Unrealized P&L: ${total_pnl:+.2f}")
                    print(f"   Total Exposure: ${sum(p['notional'] for p in current_positions.values()):.0f}")

                print("\n🧠 ML Market Analysis:")
                for coin in target_coins[:6]:
                    if coin in md:
                        price = md[coin]["price"]
                        change = md[coin]["change_24h"]
                        
                        # Quick ML analysis for display
                        symbol = f"{coin}USDT"
                        tf = get_multi_timeframe_data(symbol)
                        orderbook = get_order_book_analysis(symbol)
                        
                        try:
                            if hasattr(ai_trade_decision_ml, 'ml_model'):
                                ml_model = ai_trade_decision_ml.ml_model
                                features = ml_model.extract_features_vector(md[coin], tf, orderbook)
                                regime = ml_model.detect_regime(features)
                                ml_signal = ml_model.predict_signal(features, regime)
                                
                                confidence_emoji = "🔥" if ml_signal["confidence"] > 0.8 else "⚡" if ml_signal["confidence"] > 0.6 else "📊"
                                regime_emoji = "🚀" if regime == "trending_up" else "📉" if regime == "trending_down" else "📈" if regime == "volatile" else "↔️"
                                
                                pos_info = ""
                                if symbol in current_positions:
                                    pos = current_positions[symbol]
                                    pos_info = f" [{pos['direction']} {pos['leverage']}x: {pos['pnl']:+.1f}]"
                                    
                                print(f"   {coin}: ${price:.2f} ({change:+.1f}%) {regime_emoji}{regime.title()} ML:{ml_signal['confidence']:.2f}{confidence_emoji}{pos_info}")
                        except:
                            print(f"   {coin}: ${price:.2f} ({change:+.1f}%) [ML Analysis Error]")

                # Enhanced opportunity scanning
                prisk = calculate_portfolio_risk()
                if prisk["can_trade"]:
                    insights = get_learning_insights()
                    print(f"\n⚡ Aggressive Opportunity Scan:")
                    print(f"   Portfolio: {prisk['positions_count']}/{MAX_CONCURRENT_POSITIONS} | Available: ${prisk['available_balance']:.2f}")
                    print(f"   Margin Usage: {prisk['margin_ratio']*100:.1f}% | ML Win Rate: {insights['win_rate']:.1f}%")

                    # Get dynamic policy
                    policy = get_policy_adjustments()
                    b_thr = policy["global"]["bullish_thr"]
                    s_thr = policy["global"]["bearish_thr"]
                    explore = (random.random() < policy["global"]["exploration_rate"])

                    # Collect ML candidates
                    ml_candidates = []
                    for coin in target_coins:
                        if coin not in md:
                            continue
                        symbol = f"{coin}USDT"
                        if symbol in current_positions:
                            continue
                            
                        tf = get_multi_timeframe_data(symbol)
                        tp = ai_trade_decision_ml(coin, md, tf, insights)
                        
                        if tp and tp["direction"] != "SKIP":
                            ml_confidence = tp.get("ml_confidence", 0.5)
                            pattern_strength = tp.get("pattern_strength", 0.0)
                            
                            # Aggressive filtering - only high-confidence ML signals
                            if ml_confidence > 0.55 or pattern_strength > 0.20:
                                ml_candidates.append((coin, tp, ml_confidence, pattern_strength))

                    # Sort by combined ML score
                    ml_candidates.sort(key=lambda x: x[2] * (1 + x[3]), reverse=True)

                    max_new = min(3, MAX_CONCURRENT_POSITIONS - prisk["positions_count"])
                    executed = 0

                    for i, (coin, tp, ml_conf, pattern_str) in enumerate(ml_candidates[:max_new]):
                        print(f"\n   🎯 ML Candidate #{i+1}: {coin}")
                        print(f"      Direction: {tp['direction']} | ML Confidence: {ml_conf:.2f} | Pattern: {pattern_str:.2f}")
                        print(f"      Leverage: {tp.get('leverage', 15)}x | R/R: 1:{tp.get('tp_percentage', 0)/max(tp.get('sl_percentage', 1), 0.1):.1f}")

                        if tp["confidence"] >= MIN_CONFIDENCE_THRESHOLD:
                            # Get per-coin adjustments
                            adj = policy["per_coin"].get(coin, {"bias": 0.0, "size_mult": 1.0, "lev_mult": 1.0})

                            # Aggressive regime multipliers
                            reg = tp.get("market_regime", "unknown")
                            reg_size = {"trending_up":1.3, "trending_down":1.3, "ranging":0.9, "volatile":1.1}.get(reg,1.0)
                            reg_lev  = {"trending_up":1.2, "trending_down":1.2, "ranging":0.9, "volatile":1.0}.get(reg,1.0)

                            # ML-enhanced multipliers
                            ml_size_boost = 1.0 + (ml_conf - 0.6) * 0.8  # Up to 80% larger
                            ml_lev_boost = 1.0 + (pattern_str * 0.5)  # Up to 50% more leverage
                            
                            size_mult = adj["size_mult"] * reg_size * ml_size_boost
                            lev_mult = adj["lev_mult"] * reg_lev * ml_lev_boost

                            # More aggressive thresholds for high-confidence ML
                            use_b_thr = b_thr - (5.0 if ml_conf > 0.8 else 3.0 if explore and i == 0 else 0.0)
                            use_s_thr = s_thr + (5.0 if ml_conf > 0.8 else 3.0 if explore and i == 0 else 0.0)

                            ok = execute_ml_trade(
                                coin, tp, md[coin]["price"],
                                bullish_thr=use_b_thr,
                                bearish_thr=use_s_thr, 
                                size_mult=size_mult,
                                lev_mult=lev_mult,
                                score_bias=adj["bias"]
                            )
                            
                            if ok:
                                executed += 1
                                time.sleep(2)  # Brief pause between trades
                            else:
                                print("      ⚠️ Trade rejected by risk management")
                        else:
                            print("      ⏭️ Below confidence threshold")

                    if executed == 0:
                        print("   No ML opportunities met aggressive criteria")
                    else:
                        print(f"   🔥 Executed {executed} aggressive ML trade(s)")
                        
                else:
                    print(f"\n⚠️ Trading halted: {prisk['reason']}")

                # Enhanced summary
                total_value = calculate_portfolio_value(md)
                cost_data = get_cost_projections()
                final_risk = calculate_portfolio_risk()
                
                print("\n📈 ML Performance Summary:")
                print(f"   Total Portfolio Value: ${total_value:.2f}")
                print(f"   Active Positions: {final_risk['positions_count']}/{MAX_CONCURRENT_POSITIONS}")
                print(f"   Margin Utilization: {final_risk['margin_ratio']*100:.1f}%")
                print(f"   Daily P&L: ${final_risk.get('realized_today', 0):+.2f}")
                
                # ML-specific metrics
                if hasattr(ai_trade_decision_ml, 'ml_model'):
                    ml_model = ai_trade_decision_ml.ml_model
                    print(f"   ML Model Health: {ml_model.model_health}")
                    print(f"   Learning Rate: {ml_model.learning_rate:.6f}")
                    print(f"   Feature Buffer: {len(ml_model.feature_buffer)}/1000")
                
                print(f"   Session Costs: ${cost_data['current']['total']:.4f}")
                print("="*90)

                last_full = now

            time.sleep(QUICK_CHECK_INTERVAL)

        except KeyboardInterrupt:
            print("\n🛑 ML Bot stopped by user")
            break
        except Exception as e:
            print(f"\n❌ ML Loop error: {e}")
            time.sleep(QUICK_CHECK_INTERVAL)

# ============== Main Execution ==============
if __name__ == "__main__":
    # Start Flask dashboard
    flask_thread = threading.Thread(target=run_flask_app, daemon=True)
    flask_thread.start()
    
    # Start the ML-enhanced trading bot
    run_ml_enhanced_bot()