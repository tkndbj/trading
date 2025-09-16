# -*- coding: utf-8 -*-
import os, time, json, hmac, hashlib, math, threading, sqlite3, urllib.parse
from datetime import datetime, timedelta
from collections import deque
from decimal import Decimal, ROUND_DOWN

import requests
from flask import Flask, jsonify, send_from_directory
import numpy as np
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
_all_usdt_symbols = set()

def load_exchange_info():
    data = binance_request("/fapi/v1/exchangeInfo", "GET", signed=False)
    for s in data["symbols"]:
        sym = s["symbol"]
        if s["status"] == "TRADING" and s["contractType"] == "PERPETUAL":
            if sym.endswith("USDT"):
                _all_usdt_symbols.add(sym)
            
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

# ============== Configuration ==============
POSITION_SIZE = 0.15  # Fixed 15% of portfolio
LEVERAGE = 15  # Fixed 15x leverage
MAX_CONCURRENT_POSITIONS = 8
CHECK_INTERVAL = 30  # 30 seconds

# Favorite coins - can be manually curated for "right room" strategy
FAVORITE_COINS = {'BOME','PENDLE','JUP','LINEA','UB','ZEC','CGPT','POPCAT','WIF','OL','JASMY','BLUR','GMX','COMP','CRV','SNX','1INCH','SUSHI','YFI','BAL','MKR'}

# ENHANCEMENT: Multi-timeframe data cache
_tf_cache = {}

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
        strategy_type TEXT,
        grid_levels TEXT,
        last_checked DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS grid_positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        coin TEXT NOT NULL,
        symbol TEXT NOT NULL,
        grid_id TEXT NOT NULL,
        level_price REAL NOT NULL,
        level_type TEXT NOT NULL,
        order_id TEXT,
        filled BOOLEAN DEFAULT FALSE,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.commit()
    conn.close()
    print("Database initialized")

# ============== Market Data Functions ==============
def get_futures_balance():
    try:
        info = binance_request("/fapi/v2/account", "GET", signed=True)
        if not info: return 0.0
        total = float(info.get("totalWalletBalance", 0))
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
            if abs(amt) <= 0: 
                continue
                
            sym = p["symbol"]
            coin = sym.replace("USDT","").replace("USDC","").replace("BNFCR","")
            
            entry = float(p.get("entryPrice") or 0)
            mark = float(p.get("markPrice") or 0)
            unrealized_pnl = float(p.get("unRealizedPnl", 0))
            
            if unrealized_pnl == 0 and entry > 0 and mark > 0 and abs(amt) > 0:
                if amt > 0:
                    unrealized_pnl = amt * (mark - entry)
                else:
                    unrealized_pnl = amt * (mark - entry)
            
            leverage = int(float(p.get("leverage", "1")))
            notional = abs(amt * mark)
            
            out[sym] = {
                "coin": coin, 
                "symbol": sym, 
                "size": amt,
                "direction": "LONG" if amt > 0 else "SHORT",
                "entry_price": entry, 
                "mark_price": mark, 
                "pnl": unrealized_pnl,
                "leverage": leverage, 
                "notional": notional,
                "percentage": float(p.get("percentage", "0"))
            }
            
        return out
        
    except Exception as e:
        print(f"Positions error: {e}")
        return {}

def get_batch_market_data():
    """Fetch market data for all USDT perpetuals, filter mid-caps dynamically"""
    try:
        data = binance_request("/fapi/v1/ticker/24hr", "GET", signed=False) or []
        md = {}
        
        for t in data:
            sym = t["symbol"]
            if not sym.endswith("USDT") or sym not in _all_usdt_symbols: 
                continue
                
            coin = sym[:-4]
            quote_vol = float(t["quoteVolume"])
            
            # Dynamic filtering: mid-cap coins (10M-100M USDT) + favorites
            if quote_vol >= 10_000_000 and (quote_vol <= 100_000_000 or coin in FAVORITE_COINS):
                md[coin] = {
                    "price": float(t["lastPrice"]),
                    "change_24h": float(t["priceChangePercent"]),
                    "volume": float(t["volume"]),
                    "high_24h": float(t["highPrice"]),
                    "low_24h": float(t["lowPrice"]),
                    "quote_volume": quote_vol
                }
        
        return md
        
    except Exception as e:
        print(f"24hr fetch error: {e}")
        return {}

def get_multi_timeframe_data(symbol):
    """ENHANCED: Cached multi-timeframe data to reduce API load"""
    now = time.time()
    
    # Check cache first
    if symbol in _tf_cache and now - _tf_cache[symbol]["time"] < 60:
        return _tf_cache[symbol]["data"]
    
    intervals = {'5m':('5m',100),'15m':('15m',100),'1h':('1h',100),'4h':('4h',100)}
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
    
    # Cache the result
    _tf_cache[symbol] = {"data": out, "time": now}
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

# ============== Technical Analysis Functions ==============
def calculate_ema(prices, period):
    """ENHANCED: Calculate Exponential Moving Average with Decimal precision"""
    if len(prices) < period:
        return None
    
    # Convert to Decimal for precision
    decimal_prices = [Decimal(str(p)) for p in prices]
    alpha = Decimal('2.0') / (period + 1)
    ema = decimal_prices[0]
    
    for price in decimal_prices[1:]:
        ema = alpha * price + (Decimal('1') - alpha) * ema
    
    return float(ema)

def calculate_sma(prices, period):
    """ENHANCED: Calculate Simple Moving Average with Decimal precision"""
    if len(prices) < period:
        return None
    
    # Convert to Decimal for precision
    decimal_prices = [Decimal(str(p)) for p in prices[-period:]]
    return float(sum(decimal_prices) / period)

def calculate_rsi(prices, period=14):
    """ENHANCED: Calculate RSI with Decimal precision"""
    if len(prices) < period + 1:
        return None
    
    # Convert to Decimal for precision
    decimal_prices = [Decimal(str(p)) for p in prices]
    gains = []
    losses = []
    
    for i in range(1, len(decimal_prices)):
        change = decimal_prices[i] - decimal_prices[i-1]
        if change > 0:
            gains.append(change)
            losses.append(Decimal('0'))
        else:
            gains.append(Decimal('0'))
            losses.append(-change)
    
    if len(gains) < period:
        return None
        
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi)

def calculate_atr(highs, lows, closes, period=14, current_price=None):
    """ENHANCED: Calculate ATR with price-relative capping to prevent absurd values"""
    if len(highs) < period+1 or len(lows) < period+1 or len(closes) < period+1:
        # If no current_price provided, use a conservative default
        fallback = (current_price * 0.02) if current_price else 0.02
        return max(fallback, 0.001)
    
    tr = []
    for i in range(1, len(highs)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i-1])
        lc = abs(lows[i] - closes[i-1])
        tr.append(max(hl, hc, lc))
    
    if len(tr) < period:
        fallback = (current_price * 0.02) if current_price else 0.02
        return max(fallback, 0.001)
    
    atr_value = sum(tr[-period:]) / period
    
    # ENHANCEMENT: Cap ATR at 10% of current price to prevent absurd SL/TP
    if current_price:
        max_atr = current_price * 0.1  # Cap at 10% of price
        atr_value = min(atr_value, max_atr)
    
    return max(atr_value, 0.001)  # Ensure minimum ATR

def detect_market_regime(multi_tf_data, current_price):
    """Detect market regime: trending_up, trending_down, ranging, volatile"""
    try:
        if "1h" not in multi_tf_data or len(multi_tf_data["1h"]["closes"]) < 50:
            return "ranging"
            
        closes = multi_tf_data["1h"]["closes"]
        highs = multi_tf_data["1h"]["highs"]
        lows = multi_tf_data["1h"]["lows"]
        
        # Calculate trend strength
        ema_20 = calculate_ema(closes, 20)
        ema_50 = calculate_ema(closes, 50)
        
        if not ema_20 or not ema_50:
            return "ranging"
            
        # Trend direction
        trend_strength = (ema_20 - ema_50) / ema_50
        
        # Volatility measure
        atr = calculate_atr(highs, lows, closes, 14, current_price)
        volatility_ratio = atr / current_price if current_price > 0 else 0
        
        # Regime classification
        if volatility_ratio > 0.05:  # High volatility
            return "volatile"
        elif trend_strength > 0.02:  # Strong uptrend
            return "trending_up"
        elif trend_strength < -0.02:  # Strong downtrend
            return "trending_down"
        else:
            return "ranging"
            
    except Exception as e:
        print(f"Regime detection error: {e}")
        return "ranging"

def detect_rsi_divergence(multi_tf_data, current_price):
    """Detect RSI divergence for momentum confirmation"""
    try:
        if "1h" not in multi_tf_data or len(multi_tf_data["1h"]["closes"]) < 50:
            return False, "No data"
            
        closes = multi_tf_data["1h"]["closes"]
        highs = multi_tf_data["1h"]["highs"]
        lows = multi_tf_data["1h"]["lows"]
        
        # Calculate RSI for recent periods
        rsi_values = []
        for i in range(30, len(closes)):  # Calculate RSI for last 20 periods
            rsi = calculate_rsi(closes[:i+1], 14)
            if rsi is not None:
                rsi_values.append(rsi)
        
        if len(rsi_values) < 10:
            return False, "Insufficient RSI data"
        
        # Look for divergence in last 10 periods
        recent_highs = highs[-10:]
        recent_lows = lows[-10:]
        recent_rsi = rsi_values[-10:]
        
        # Bullish divergence: price makes lower lows, RSI makes higher lows
        price_ll = min(recent_lows) == recent_lows[-1] and recent_lows[-1] < recent_lows[0]
        rsi_hl = min(recent_rsi) != recent_rsi[-1] and recent_rsi[-1] > min(recent_rsi)
        
        # Bearish divergence: price makes higher highs, RSI makes lower highs  
        price_hh = max(recent_highs) == recent_highs[-1] and recent_highs[-1] > recent_highs[0]
        rsi_lh = max(recent_rsi) != recent_rsi[-1] and recent_rsi[-1] < max(recent_rsi)
        
        if price_ll and rsi_hl:
            return True, "Bullish RSI divergence"
        elif price_hh and rsi_lh:
            return True, "Bearish RSI divergence"
        else:
            return False, "No divergence"
            
    except Exception as e:
        print(f"RSI divergence error: {e}")
        return False, "Error"

def detect_breakout(multi_tf_data, current_price, volume_data):
    """Detect breakouts above resistance with volume confirmation and RSI divergence"""
    try:
        if "1h" not in multi_tf_data or len(multi_tf_data["1h"]["closes"]) < 50:
            return False, "No data"
            
        closes = multi_tf_data["1h"]["closes"]
        highs = multi_tf_data["1h"]["highs"]
        volumes = multi_tf_data["1h"]["volumes"]
        
        # Find recent resistance level (highest high in last 20 periods)
        resistance = max(highs[-20:])
        
        # Check if current price is breaking above resistance
        breakout_threshold = resistance * 1.002  # 0.2% above resistance
        is_breakout = current_price > breakout_threshold
        
        if not is_breakout:
            return False, "No breakout"
            
        # Volume confirmation
        avg_volume = sum(volumes[-20:]) / 20
        recent_volume = volumes[-1]
        volume_surge = recent_volume > avg_volume * 1.5  # 50% above average
        
        # Price momentum confirmation
        price_momentum = (current_price - closes[-5]) / closes[-5] > 0.02  # 2% move
        
        # RSI divergence confirmation (ADDED)
        has_divergence, div_reason = detect_rsi_divergence(multi_tf_data, current_price)
        
        confirmations = []
        if volume_surge:
            confirmations.append("volume")
        if price_momentum:
            confirmations.append("momentum")
        if has_divergence:
            confirmations.append(div_reason)
        
        if len(confirmations) >= 2:  # Need at least 2 confirmations
            return True, f"Breakout above {resistance:.4f} ({', '.join(confirmations)})"
        elif len(confirmations) == 1:
            return True, f"Weak breakout above {resistance:.4f} ({confirmations[0]})"
        else:
            return False, "Breakout lacks confirmation"
            
    except Exception as e:
        print(f"Breakout detection error: {e}")
        return False, "Error"

def identify_high_volume_coins(market_data, min_volume_usdt=10_000_000):
    """Identify coins with high trading volume - now includes all filtered coins"""
    high_volume_coins = []
    for coin, data in market_data.items():
        if data["quote_volume"] >= min_volume_usdt:
            volume_score = data["quote_volume"] / 50_000_000  # Normalize to 50M (mid-cap sweet spot)
            high_volume_coins.append((coin, volume_score, data))
    
    # Sort by volume score
    high_volume_coins.sort(key=lambda x: x[1], reverse=True)
    return high_volume_coins[:20]  # Top 20 by volume

def detect_supply_demand_zones(multi_tf_data):
    """Detect supply and demand zones"""
    try:
        if "1h" not in multi_tf_data or len(multi_tf_data["1h"]["closes"]) < 50:
            return {"demand_zone": None, "supply_zone": None}
            
        highs = multi_tf_data["1h"]["highs"]
        lows = multi_tf_data["1h"]["lows"]
        closes = multi_tf_data["1h"]["closes"]
        volumes = multi_tf_data["1h"]["volumes"]
        
        # Find significant volume spikes
        avg_volume = sum(volumes[-30:]) / 30
        volume_threshold = avg_volume * 2
        
        demand_zones = []
        supply_zones = []
        
        for i in range(len(volumes) - 10, len(volumes)):
            if volumes[i] > volume_threshold:
                # High volume area - potential S/D zone
                if closes[i] > closes[i-1]:  # Bullish volume spike
                    demand_zones.append(lows[i])
                else:  # Bearish volume spike
                    supply_zones.append(highs[i])
        
        # Get most recent zones
        demand_zone = max(demand_zones) if demand_zones else None
        supply_zone = min(supply_zones) if supply_zones else None
        
        return {"demand_zone": demand_zone, "supply_zone": supply_zone}
        
    except Exception as e:
        print(f"S/D zone detection error: {e}")
        return {"demand_zone": None, "supply_zone": None}

def calculate_mean_reversion_signal(multi_tf_data, current_price):
    """Calculate mean reversion signal strength"""
    try:
        if "1h" not in multi_tf_data or len(multi_tf_data["1h"]["closes"]) < 50:
            return 0, "No data"
            
        closes = multi_tf_data["1h"]["closes"]
        
        # Calculate Bollinger Bands
        sma_20 = calculate_sma(closes, 20)
        if not sma_20:
            return 0, "No SMA"
            
        # Standard deviation
        squared_diffs = [(price - sma_20) ** 2 for price in closes[-20:]]
        std_dev = (sum(squared_diffs) / 20) ** 0.5
        
        upper_band = sma_20 + (2 * std_dev)
        lower_band = sma_20 - (2 * std_dev)
        
        # Calculate position relative to bands
        if current_price > upper_band:
            # Overbought - bearish mean reversion
            distance = (current_price - upper_band) / upper_band
            return -min(1.0, distance * 10), "Overbought"
        elif current_price < lower_band:
            # Oversold - bullish mean reversion
            distance = (lower_band - current_price) / lower_band
            return min(1.0, distance * 10), "Oversold"
        else:
            return 0, "Neutral"
            
    except Exception as e:
        print(f"Mean reversion error: {e}")
        return 0, "Error"

def calculate_momentum_signal(multi_tf_data):
    """Calculate momentum signal strength"""
    try:
        if "15m" not in multi_tf_data or len(multi_tf_data["15m"]["closes"]) < 30:
            return 0, "No data"
            
        closes = multi_tf_data["15m"]["closes"]
        
        # Price momentum (last 5 vs previous 10)
        recent_avg = sum(closes[-5:]) / 5
        previous_avg = sum(closes[-15:-5]) / 10
        
        momentum = (recent_avg - previous_avg) / previous_avg
        
        # Normalize to [-1, 1]
        momentum_signal = max(-1.0, min(1.0, momentum * 20))
        
        return momentum_signal, f"Momentum: {momentum:.4f}"
        
    except Exception as e:
        print(f"Momentum error: {e}")
        return 0, "Error"

# ============== Grid Trading System ==============
def create_grid_levels(current_price, range_pct=0.05, levels=10):
    """Create grid levels for ranging market"""
    range_size = current_price * range_pct
    level_spacing = range_size / levels
    
    grid_levels = []
    for i in range(-levels//2, levels//2 + 1):
        level_price = current_price + (i * level_spacing)
        level_type = "BUY" if i < 0 else "SELL" if i > 0 else "CENTER"
        grid_levels.append({
            "price": level_price,
            "type": level_type,
            "level": i
        })
    
    return grid_levels

def setup_grid_trading(coin, symbol, current_price, balance):
    """Setup grid trading for ranging market"""
    try:
        grid_levels = create_grid_levels(current_price)
        grid_id = f"GRID_{symbol}_{int(time.time())}"
        
        # Store grid configuration
        for level in grid_levels:
            db_execute("""
                INSERT INTO grid_positions (coin, symbol, grid_id, level_price, level_type)
                VALUES (?, ?, ?, ?, ?)
            """, (coin, symbol, grid_id, level["price"], level["type"]))
        
        print(f"Grid setup for {coin}: {len(grid_levels)} levels around ${current_price:.4f}")
        return grid_id
        
    except Exception as e:
        print(f"Grid setup error: {e}")
        return None

def check_grid_triggers(symbol, current_price):
    """Check if any grid levels should be triggered"""
    try:
        # Get unfilled grid levels
        levels = db_execute("""
            SELECT id, level_price, level_type, grid_id
            FROM grid_positions 
            WHERE symbol = ? AND filled = FALSE
        """, (symbol,), fetch=True)
        
        triggered_levels = []
        for level_id, level_price, level_type, grid_id in levels:
            if level_type == "BUY" and current_price <= level_price:
                triggered_levels.append((level_id, level_price, "BUY", grid_id))
            elif level_type == "SELL" and current_price >= level_price:
                triggered_levels.append((level_id, level_price, "SELL", grid_id))
        
        return triggered_levels
        
    except Exception as e:
        print(f"Grid trigger check error: {e}")
        return []

# ============== Trading Decision Engine ==============
def get_trade_decision(coin, market_data, multi_tf_data):
    """ENHANCED: Main trading decision logic with order book analysis"""
    try:
        current_price = market_data[coin]["price"]
        symbol = f"{coin}USDT"
        
        # 1. Detect market regime
        regime = detect_market_regime(multi_tf_data, current_price)
        
        # 2. High volume filter (now more lenient due to mid-cap focus)
        if market_data[coin]["quote_volume"] < 10_000_000:  # Minimum 10M USDT volume
            return {"direction": "SKIP", "reason": "Low volume"}
        
        # ENHANCEMENT 1: Order book liquidity check - critical for mid-caps
        ob_data = get_order_book_analysis(symbol)
        if ob_data["spread"] > 0.1 or ob_data["liquidity_score"] < 40:
            return {
                "direction": "SKIP", 
                "reason": f"Illiquid: spread={ob_data['spread']:.2f}%, liq_score={ob_data['liquidity_score']:.1f}"
            }
        
        # 3. Supply/Demand analysis
        sd_zones = detect_supply_demand_zones(multi_tf_data)
        
        # 4. Different strategies based on regime
        if regime == "ranging":
            # Grid trading for ranging markets
            return {"direction": "GRID", "reason": "Ranging market - use grid", "regime": regime}
            
        elif regime in ["trending_up", "trending_down"]:
            # Breakout strategy for trending markets
            is_breakout, breakout_reason = detect_breakout(multi_tf_data, current_price, market_data[coin])
            
            if is_breakout:
                direction = "LONG" if regime == "trending_up" else "SHORT"
                
                # Calculate stops based on ATR with price awareness
                atr = calculate_atr(
                    multi_tf_data["1h"]["highs"], 
                    multi_tf_data["1h"]["lows"], 
                    multi_tf_data["1h"]["closes"],
                    14,
                    current_price  # Pass current price for capping
                )
                
                if direction == "LONG":
                    stop_loss = current_price - (atr * 2)
                    take_profit = current_price + (atr * 4)
                else:
                    stop_loss = current_price + (atr * 2)
                    take_profit = current_price - (atr * 4)
                
                return {
                    "direction": direction,
                    "reason": f"Breakout in {regime} market: {breakout_reason}",
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "regime": regime,
                    "strategy": "breakout",
                    "liquidity": f"Spread: {ob_data['spread']:.3f}%, Liquidity: {ob_data['liquidity_score']:.1f}"
                }
                
        elif regime == "volatile":
            # Mean reversion + momentum hybrid for volatile markets
            mean_rev_signal, mean_rev_reason = calculate_mean_reversion_signal(multi_tf_data, current_price)
            momentum_signal, momentum_reason = calculate_momentum_signal(multi_tf_data)
            
            # Combine signals (momentum gets 60% weight, mean reversion 40%)
            combined_signal = (momentum_signal * 0.6) + (mean_rev_signal * 0.4)
            
            if abs(combined_signal) > 0.6:  # Strong signal threshold
                direction = "LONG" if combined_signal > 0 else "SHORT"
                
                # Calculate stops with price awareness
                atr = calculate_atr(
                    multi_tf_data["1h"]["highs"], 
                    multi_tf_data["1h"]["lows"], 
                    multi_tf_data["1h"]["closes"],
                    14,
                    current_price  # Pass current price for capping
                )
                
                if direction == "LONG":
                    stop_loss = current_price - (atr * 1.5)
                    take_profit = current_price + (atr * 3)
                else:
                    stop_loss = current_price + (atr * 1.5)
                    take_profit = current_price - (atr * 3)
                
                return {
                    "direction": direction,
                    "reason": f"Hybrid signal: {momentum_reason} + {mean_rev_reason}",
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "regime": regime,
                    "strategy": "hybrid",
                    "signal_strength": abs(combined_signal),
                    "liquidity": f"Spread: {ob_data['spread']:.3f}%, Liquidity: {ob_data['liquidity_score']:.1f}"
                }
        
        return {"direction": "SKIP", "reason": f"No signal in {regime} market"}
        
    except Exception as e:
        print(f"Decision error for {coin}: {e}")
        return {"direction": "SKIP", "reason": f"Error: {e}"}

# ============== Position Management ==============
def place_futures_order(symbol, side, quantity, leverage=LEVERAGE, order_type="MARKET"):
    try:
        _ = binance_request("/fapi/v1/leverage","POST",params={"symbol":symbol,"leverage":leverage},signed=True)
        cid = f"tech-bot-{int(time.time()*1000)}-{side}"
        params = {"symbol":symbol,"side":side,"type":order_type,"quantity":quantity,"newClientOrderId":cid}
        res = binance_request("/fapi/v1/order","POST",params=params,signed=True)
        return res
    except Exception as e:
        print(f"Place order error: {e}")
        return None

def close_futures_position(symbol, partial_qty=None):
    """Close position - can now handle partial closes"""
    try:
        positions = get_open_positions()
        if symbol not in positions: 
            print(f"No open position for {symbol}")
            return None
        pos = positions[symbol]
        
        if partial_qty is not None:
            # Partial close
            size = q_qty(symbol, partial_qty)
        else:
            # Full close
            size = abs(pos["size"])
            
        side = "SELL" if pos["direction"]=="LONG" else "BUY"
        cid = f"tech-close-{int(time.time()*1000)}"
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

def cancel_all_orders(symbol):
    """Cancel all open orders for a symbol"""
    try:
        res = binance_request("/fapi/v1/allOpenOrders","DELETE",params={"symbol":symbol},signed=True)
        return res
    except Exception as e:
        print(f"Cancel orders error: {e}")
        return None

def move_stop_to_entry(symbol, entry_price):
    """Move stop loss to entry price (breakeven)"""
    try:
        positions = get_open_positions()
        if symbol not in positions:
            return None
            
        pos = positions[symbol]
        
        # Cancel existing SL/TP orders
        cancel_all_orders(symbol)
        
        # Set new SL at entry
        qty = abs(pos["size"])
        qty = q_qty(symbol, qty)
        sl_side = "SELL" if pos["direction"] == "LONG" else "BUY"
        
        sl_order = binance_request("/fapi/v1/order","POST",signed=True,params={
            "symbol": symbol, "side": sl_side, "type": "STOP_MARKET", "quantity": qty,
            "stopPrice": q_price(symbol, entry_price), "workingType": "MARK_PRICE", "reduceOnly": True
        })
        
        return sl_order
        
    except Exception as e:
        print(f"Move SL error: {e}")
        return None

def execute_trade(coin, trade_params, current_price, balance):
    """Execute trade with live logging - FIXED POSITION SIZING"""
    if trade_params["direction"] in ["SKIP", "GRID"]:
        return False
        
    try:
        symbol = f"{coin}USDT"
        
        # FIXED: Position sizing with correct leverage application
        position_value = balance * POSITION_SIZE  # 15% of balance as margin
        notional = position_value * LEVERAGE      # Full exposure (15% * 15x = 225% of balance)
        qty = q_qty(symbol, notional / current_price)  # FIXED: Use notional, not position_value
        
        if qty <= 0:
            add_bot_log("ERROR", "TRADE", f"{coin}: Quantity rounded to zero")
            return False

        # Minimum notional check - FIXED: Use notional instead of position_value * current_price
        min_notional = _symbol_info.get(symbol, {}).get("minNotional", Decimal("5"))
        computed_notional = Decimal(str(notional))  # FIXED: Use actual notional
        if computed_notional < min_notional:
            add_bot_log("ERROR", "TRADE", f"{coin}: Below MIN_NOTIONAL: {float(computed_notional):.2f} < {float(min_notional):.2f}")
            return False

        # Log liquidity info if available
        liquidity_info = trade_params.get("liquidity", "")
        add_bot_log("INFO", "EXECUTE", f"{coin} {trade_params['direction']} - Margin: ${position_value:.0f} | Exposure: ${notional:.0f} ({LEVERAGE}x) - Strategy: {trade_params.get('strategy', 'unknown')} | {liquidity_info}")

        # Execute the trade
        side = "BUY" if trade_params["direction"] == "LONG" else "SELL"
        res = place_futures_order(symbol, side, qty, LEVERAGE)
        
        if not res:
            add_bot_log("ERROR", "ORDER", f"{coin}: Order execution failed")
            return False

        # Set stop loss and take profit
        if "stop_loss" in trade_params and "take_profit" in trade_params:
            sltp = set_stop_loss_take_profit(
                symbol, 
                trade_params["stop_loss"], 
                trade_params["take_profit"], 
                trade_params["direction"]
            )
            if not sltp:
                add_bot_log("ERROR", "SLTP", f"{coin}: SL/TP failed, closing position")
                close_futures_position(symbol)
                return False
            else:
                add_bot_log("INFO", "SLTP", f"{coin}: SL: ${trade_params['stop_loss']:.4f} | TP: ${trade_params['take_profit']:.4f}")

        # Store trade record
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pid = f"{symbol}_{ts.replace(' ','_').replace(':','-')}"
        
        db_execute("""
            INSERT OR REPLACE INTO active_positions (
                position_id, coin, symbol, direction, entry_price, entry_time,
                position_size, leverage, notional_value, stop_loss, take_profit,
                strategy_type
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            pid, coin, symbol, trade_params["direction"], current_price, ts,
            position_value, LEVERAGE, float(computed_notional), 
            trade_params.get("stop_loss", 0), trade_params.get("take_profit", 0),
            trade_params.get("strategy", "technical")
        ))

        db_execute("""
            INSERT INTO trades (timestamp,position_id,coin,action,direction,price,position_size,leverage,
            notional_value,stop_loss,take_profit,reason,market_conditions)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (ts, pid, coin, f"{trade_params['direction']} OPEN", trade_params["direction"], 
              current_price, position_value, LEVERAGE, float(computed_notional), 
              trade_params.get("stop_loss", 0), trade_params.get("take_profit", 0),
              trade_params["reason"], trade_params.get("regime", "unknown")))

        add_bot_log("SUCCESS", "TRADE", f"{coin} {trade_params['direction']} executed - Entry: ${current_price:.4f} | Exposure: ${notional:.0f}")
        return True

    except Exception as e:
        add_bot_log("ERROR", "TRADE", f"{coin}: Execution error - {e}")
        return False

def handle_grid_trading(coin, symbol, current_price, balance):
    """ENHANCED: Handle grid trading execution with improved level handling and grid reset"""
    try:
        # ENHANCEMENT 2: Check if grid needs reset (price out of range)
        levels = db_execute("""
            SELECT MIN(level_price), MAX(level_price) FROM grid_positions 
            WHERE symbol = ? AND filled = FALSE
        """, (symbol,), fetch=True)
        
        if levels and levels[0] and levels[0][0] is not None and levels[0][1] is not None:
            min_price, max_price = levels[0]
            # If price is more than 10% outside grid range, reset grid
            if current_price < min_price * 0.9 or current_price > max_price * 1.1:
                db_execute("DELETE FROM grid_positions WHERE symbol = ? AND filled = FALSE", (symbol,))
                add_bot_log("INFO", "GRID", f"{coin}: Reset grid - price out of range (${current_price:.4f} vs ${min_price:.4f}-${max_price:.4f})")
                # Create new grid
                grid_id = setup_grid_trading(coin, symbol, current_price, balance)
                if not grid_id:
                    return False
        
        # Check if grid exists, create if not
        existing_grid = db_execute("""
            SELECT COUNT(*) FROM grid_positions WHERE symbol = ? AND filled = FALSE
        """, (symbol,), fetch=True)
        
        if existing_grid[0][0] == 0:
            # Create new grid
            grid_id = setup_grid_trading(coin, symbol, current_price, balance)
            if not grid_id:
                return False
        
        # ENHANCEMENT 2: Check for triggered levels - now allow up to 2 levels per check
        triggered_levels = check_grid_triggers(symbol, current_price)
        
        if triggered_levels:
            # Execute up to 2 levels instead of just 1
            executed_count = 0
            for level_id, level_price, level_type, grid_id in triggered_levels[:2]:
                # Calculate position size for grid (smaller than normal trades)
                grid_position_value = balance * 0.05  # 5% per grid level
                notional = grid_position_value * LEVERAGE
                qty = q_qty(symbol, notional / current_price)
                
                if qty > 0:
                    side = "BUY" if level_type == "BUY" else "SELL"
                    res = place_futures_order(symbol, side, qty, LEVERAGE)
                    
                    if res:
                        # Mark level as filled
                        db_execute("""
                            UPDATE grid_positions SET filled = TRUE, order_id = ?
                            WHERE id = ?
                        """, (res.get("orderId", ""), level_id))
                        
                        add_bot_log("SUCCESS", "GRID", f"Grid {level_type} executed for {coin} at ${level_price:.4f}")
                        executed_count += 1
                        time.sleep(1)  # Brief delay between executions
            
            return executed_count > 0
        
        return False
        
    except Exception as e:
        add_bot_log("ERROR", "GRID", f"Grid trading error: {e}")
        return False

# ============== Position Monitoring ==============
def monitor_positions():
    """Monitor positions for regime changes and profit taking - ENHANCED with partial closes"""
    try:
        positions = get_open_positions()
        
        if positions:
            add_bot_log("INFO", "MONITOR", f"Monitoring {len(positions)} active position(s)")
        
        for symbol, pos in positions.items():
            # Get fresh market data
            multi_tf_data = get_multi_timeframe_data(symbol)
            current_regime = detect_market_regime(multi_tf_data, pos["mark_price"])
            
            # Log position status
            pnl_pct = (pos["pnl"] / (pos["notional"]/pos["leverage"])) * 100 if pos["notional"] > 0 else 0
            add_bot_log("DEBUG", "POSITION", f"{pos['coin']}: {pos['direction']} | P&L: {pnl_pct:+.1f}% | Regime: {current_regime}")
            
            # Check for regime change exit
            should_close = False
            should_partial_close = False
            should_move_sl = False
            close_reason = ""
            
            # Get position details from database
            pos_details = db_execute("""
                SELECT strategy_type FROM active_positions WHERE symbol = ?
            """, (symbol,), fetch=True)
            
            if pos_details:
                strategy = pos_details[0][0]
                
                # ENHANCED: Profit management
                if pnl_pct >= 100:  # 100% profit - close completely
                    should_close = True
                    close_reason = f"100% profit target reached ({pnl_pct:.1f}%)"
                elif pnl_pct >= 50:  # 50% profit - close half and move SL to entry
                    should_partial_close = True
                    should_move_sl = True
                    close_reason = f"Partial close at 50% profit ({pnl_pct:.1f}%)"
                elif pnl_pct > 8:  # Old profit taking logic (reduced threshold)
                    should_close = True
                    close_reason = f"Profit taking at {pnl_pct:.1f}%"
                elif pnl_pct < -20:  # CHANGED: Emergency stop from -4% to -20%
                    should_close = True
                    close_reason = f"Emergency stop at {pnl_pct:.1f}%"
                
                # Exit conditions based on strategy
                if strategy == "breakout" and not should_close and not should_partial_close:
                    if current_regime in ["ranging", "volatile"]:
                        should_close = True
                        close_reason = f"Regime changed to {current_regime}"
                        
                elif strategy == "hybrid" and not should_close and not should_partial_close:
                    momentum_signal, _ = calculate_momentum_signal(multi_tf_data)
                    position_direction = 1 if pos["direction"] == "LONG" else -1
                    
                    if momentum_signal * position_direction < -0.3:
                        should_close = True
                        close_reason = "Momentum reversal detected"
            
            # Execute actions
            if should_partial_close:
                # Close half position
                half_size = abs(pos["size"]) / 2
                add_bot_log("WARNING", "PARTIAL", f"Partial close {pos['coin']}: {close_reason}")
                res = close_futures_position(symbol, half_size)
                
                if res and should_move_sl:
                    # Move SL to entry
                    move_res = move_stop_to_entry(symbol, pos["entry_price"])
                    if move_res:
                        add_bot_log("INFO", "BREAKEVEN", f"{pos['coin']}: Stop moved to breakeven")
                    
                    # Record the partial closure
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    pnl_usd = pos["pnl"] / 2  # Rough estimate for half position
                    pnl_percent = pnl_pct
                    
                    db_execute("""
                        INSERT INTO trades (timestamp, position_id, coin, action, direction, price,
                                          pnl, pnl_percent, reason, profitable)
                        VALUES (?,?,?,?,?,?,?,?,?,?)
                    """, (ts, f"{symbol}_PARTIAL_{ts.replace(' ','_').replace(':','-')}",
                          pos["coin"], f"{pos['direction']} PARTIAL", pos["direction"], 
                          pos["mark_price"], pnl_usd, pnl_percent, close_reason, 
                          1 if pnl_usd > 0 else 0))
                    
                    add_bot_log("SUCCESS", "PARTIAL", f"Partial close {pos['coin']}: P&L ${pnl_usd:+.2f} ({pnl_percent:+.1f}%)")
                    
            elif should_close:
                add_bot_log("WARNING", "EXIT", f"Closing {pos['coin']}: {close_reason}")
                res = close_futures_position(symbol)
                
                if res:
                    # Record the closure
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    pnl_usd = pos["pnl"]
                    pnl_percent = (pnl_usd / (pos["notional"]/pos["leverage"])) * 100 if pos["notional"] > 0 else 0
                    
                    db_execute("""
                        INSERT INTO trades (timestamp, position_id, coin, action, direction, price,
                                          pnl, pnl_percent, reason, profitable)
                        VALUES (?,?,?,?,?,?,?,?,?,?)
                    """, (ts, f"{symbol}_CLOSE_{ts.replace(' ','_').replace(':','-')}",
                          pos["coin"], f"{pos['direction']} CLOSE", pos["direction"], 
                          pos["mark_price"], pnl_usd, pnl_percent, close_reason, 
                          1 if pnl_usd > 0 else 0))
                    
                    # Remove from active positions
                    db_execute("DELETE FROM active_positions WHERE symbol = ?", (symbol,))
                    
                    add_bot_log("SUCCESS", "CLOSE", f"Closed {pos['coin']}: P&L ${pnl_usd:+.2f} ({pnl_percent:+.1f}%)")
                    
        return positions
        
    except Exception as e:
        add_bot_log("ERROR", "MONITOR", f"Monitor error: {e}")
        return {}

def calculate_portfolio_stats(market_data):
    """Calculate portfolio statistics"""
    balance = get_futures_balance()
    positions = get_open_positions()
    total_pnl = sum(p["pnl"] for p in positions.values())
    total_value = balance + total_pnl
    
    # Calculate daily P&L
    today_pnl = db_execute("""
        SELECT COALESCE(SUM(pnl), 0) FROM trades
        WHERE action LIKE '%CLOSE%' AND DATE(timestamp) = DATE('now')
    """, fetch=True)[0][0]
    
    return {
        "balance": balance,
        "unrealized_pnl": total_pnl,
        "total_value": total_value,
        "daily_pnl": today_pnl,
        "positions_count": len(positions),
        "total_exposure": sum(p["notional"] for p in positions.values())
    }

# ============== Flask API ==============
app = Flask(__name__)

@app.route("/")
def dashboard():
    return send_from_directory(".", "index.html")

@app.route("/api/status")
def api_status():
    try:
        md = get_batch_market_data()
        positions = get_open_positions()
        stats = calculate_portfolio_stats(md)
        
        # Get recent trades - fixed ordering
        recent_trades = db_execute("""
            SELECT timestamp, coin, action, direction, price, pnl, pnl_percent, reason
            FROM trades 
            ORDER BY created_at DESC, timestamp DESC 
            LIMIT 20
        """, fetch=True)
        
        trade_history = []
        for trade in recent_trades:
            trade_history.append({
                "timestamp": trade[0],
                "coin": trade[1],
                "action": trade[2],
                "direction": trade[3],
                "price": trade[4],
                "pnl": trade[5],
                "pnl_percent": trade[6],
                "reason": trade[7]
            })
        
        # Get live bot logs from the last hour
        logs = get_recent_bot_logs()
        
        return jsonify({
            "portfolio_stats": stats,
            "positions": positions,
            "market_data": md,
            "trade_history": trade_history,
            "bot_logs": logs,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Add this new logging system
bot_logs = deque(maxlen=100)  # Keep last 100 log entries

def add_bot_log(level, component, message):
    """Add a log entry that will be shown in the dashboard"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "level": level,
        "component": component,
        "message": message
    }
    bot_logs.append(log_entry)
    print(f"[{timestamp}] [{level}] [{component}] {message}")

def get_recent_bot_logs():
    """Get recent bot logs for the dashboard"""
    return list(bot_logs)

def run_flask_app():
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)

# ============== Main Trading Loop ==============
def run_technical_bot():
    """Main trading loop with technical analysis - Updated with live logging"""
    init_database()
    sync_server_time()
    load_exchange_info()

    add_bot_log("INFO", "STARTUP", "Technical Analysis Trading Bot starting...")
    add_bot_log("INFO", "CONFIG", f"Position Size: {POSITION_SIZE*100}% | Leverage: {LEVERAGE}x | Max Positions: {MAX_CONCURRENT_POSITIONS}")
    add_bot_log("INFO", "TARGETS", f"Tracking {len(_all_usdt_symbols)} USDT perpetuals | Favorites: {len(FAVORITE_COINS)} coins")
    add_bot_log("INFO", "ENHANCEMENTS", "✓ Order book analysis ✓ Grid improvements ✓ ATR capping ✓ Decimal precision ✓ API caching")
    
    balance = get_futures_balance()
    if balance <= 0:
        add_bot_log("ERROR", "BALANCE", "No futures balance or API error")
        return
        
    add_bot_log("INFO", "BALANCE", f"Connected | Balance: ${balance:.2f}")

    iteration = 0
    last_scan = 0

    while True:
        try:
            iteration += 1
            now = time.time()
            
            # Monitor existing positions
            positions = monitor_positions()
            
            # Full market scan every 3 minutes
            if now - last_scan >= 180:
                add_bot_log("INFO", "SCAN", f"Market Scan #{iteration} starting...")
                
                balance = get_futures_balance()
                md = get_batch_market_data()
                stats = calculate_portfolio_stats(md)
                
                add_bot_log("INFO", "PORTFOLIO", f"Value: ${stats['total_value']:.2f} | Positions: {stats['positions_count']}/{MAX_CONCURRENT_POSITIONS} | Daily P&L: ${stats['daily_pnl']:+.2f}")
                
                # Identify high volume candidates
                high_volume_coins = identify_high_volume_coins(md)
                add_bot_log("INFO", "ANALYSIS", f"Found {len(high_volume_coins)} mid-cap candidates | Universe: {len(md)} coins | Cache: {len(_tf_cache)} entries")
                
                # Check if we can open new positions
                if stats['positions_count'] < MAX_CONCURRENT_POSITIONS:
                    max_new = MAX_CONCURRENT_POSITIONS - stats['positions_count']
                    executed = 0
                    skipped_liquidity = 0
                    
                    for coin, volume_score, coin_data in high_volume_coins[:20]:
                        if executed >= max_new:
                            break
        
                        symbol = f"{coin}USDT"
                        if symbol in positions:
                            continue
        
                        # Get technical analysis - THIS IS MISSING!
                        add_bot_log("INFO", "ANALYSIS", f"Analyzing {coin} - Price: ${coin_data['price']:.4f} | Volume: {coin_data['quote_volume']/1000000:.1f}M | Mid-cap: {'✓' if coin_data['quote_volume'] <= 100_000_000 else '✗'}")
    
                        multi_tf_data = get_multi_timeframe_data(symbol)
                        trade_decision = get_trade_decision(coin, md, multi_tf_data)
                        if trade_decision["direction"] == "SKIP" and "Illiquid" in trade_decision["reason"]:
                            skipped_liquidity += 1
                            add_bot_log("DEBUG", "LIQUIDITY", f"{coin}: {trade_decision['reason']}")
                            continue
                        
                        if trade_decision["direction"] == "GRID":
                            add_bot_log("INFO", "STRATEGY", f"{coin}: {trade_decision['reason']}")
                            if handle_grid_trading(coin, symbol, coin_data["price"], balance):
                                executed += 1
                                add_bot_log("SUCCESS", "GRID", f"Grid setup complete for {coin}")
                                
                        elif trade_decision["direction"] in ["LONG", "SHORT"]:
                            add_bot_log("INFO", "SIGNAL", f"{coin} {trade_decision['direction']}: {trade_decision['reason']}")
                            if execute_trade(coin, trade_decision, coin_data["price"], balance):
                                executed += 1
                                add_bot_log("SUCCESS", "TRADE", f"{coin} {trade_decision['direction']} executed successfully")
                                time.sleep(2)
                        elif trade_decision["direction"] == "SKIP":
                            add_bot_log("DEBUG", "SKIP", f"{coin}: {trade_decision['reason']}")
                    
                    if executed == 0:
                        reason_msg = f" ({skipped_liquidity} illiquid)" if skipped_liquidity > 0 else ""
                        add_bot_log("INFO", "SCAN", f"No trading opportunities found in this scan{reason_msg}")
                    else:
                        add_bot_log("SUCCESS", "SCAN", f"Executed {executed} new position(s)")
                        
                else:
                    add_bot_log("WARNING", "LIMIT", "Maximum positions reached - monitoring only")
                
                last_scan = now
                
            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            add_bot_log("INFO", "SHUTDOWN", "Bot stopped by user")
            break
        except Exception as e:
            add_bot_log("ERROR", "SYSTEM", f"Main loop error: {e}")
            time.sleep(CHECK_INTERVAL)

# ============== Main Execution ==============
if __name__ == "__main__":
    # Start Flask dashboard
    flask_thread = threading.Thread(target=run_flask_app, daemon=True)
    flask_thread.start()
    
    # Start the technical trading bot
    run_technical_bot()