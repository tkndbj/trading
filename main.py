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

# ============== Configuration ==============
POSITION_SIZE = 0.15  # Fixed 15% of portfolio
LEVERAGE = 15  # Fixed 15x leverage
MAX_CONCURRENT_POSITIONS = 8
CHECK_INTERVAL = 30  # 30 seconds

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
    try:
        data = binance_request("/fapi/v1/ticker/24hr", "GET", signed=False) or []
        target = {'BTC','ETH','SOL','BNB','SEI','DOGE','TIA','TAO','ARB','SUI','ENA','FET','APT','ZK','OP','LDO','WLD','XRP','LINK','AVAX','MATIC','ADA','DOT','UNI','ATOM','NEAR','ICP','FIL','LTC','BCH','ETC','HBAR','VET','ALGO','THETA'}
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
    """Calculate Exponential Moving Average"""
    if len(prices) < period:
        return None
    alpha = 2.0 / (period + 1.0)
    ema = prices[0]
    for price in prices[1:]:
        ema = alpha * price + (1 - alpha) * ema
    return ema

def calculate_sma(prices, period):
    """Calculate Simple Moving Average"""
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period

def calculate_atr(highs, lows, closes, period=14):
    """Calculate Average True Range"""
    if len(highs) < period+1 or len(lows) < period+1 or len(closes) < period+1:
        return 0
    tr = []
    for i in range(1, len(highs)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i-1])
        lc = abs(lows[i] - closes[i-1])
        tr.append(max(hl, hc, lc))
    if len(tr) < period:
        return 0
    return sum(tr[-period:]) / period

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
        atr = calculate_atr(highs, lows, closes, 14)
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

def detect_breakout(multi_tf_data, current_price, volume_data):
    """Detect breakouts above resistance with volume confirmation"""
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
        
        if volume_surge and price_momentum:
            return True, f"Breakout above {resistance:.4f} with volume"
        elif volume_surge:
            return True, f"Breakout above {resistance:.4f} (volume confirmed)"
        elif price_momentum:
            return True, f"Breakout above {resistance:.4f} (momentum confirmed)"
        else:
            return False, "Weak breakout signal"
            
    except Exception as e:
        print(f"Breakout detection error: {e}")
        return False, "Error"

def identify_high_volume_coins(market_data, min_volume_usdt=50_000_000):
    """Identify coins with high trading volume"""
    high_volume_coins = []
    for coin, data in market_data.items():
        if data["quote_volume"] > min_volume_usdt:
            volume_score = data["quote_volume"] / 100_000_000  # Normalize to 100M
            high_volume_coins.append((coin, volume_score, data))
    
    # Sort by volume score
    high_volume_coins.sort(key=lambda x: x[1], reverse=True)
    return high_volume_coins[:15]  # Top 15 by volume

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
    """Main trading decision logic"""
    try:
        current_price = market_data[coin]["price"]
        
        # 1. Detect market regime
        regime = detect_market_regime(multi_tf_data, current_price)
        
        # 2. High volume filter
        if market_data[coin]["quote_volume"] < 30_000_000:  # Minimum 30M USDT volume
            return {"direction": "SKIP", "reason": "Low volume"}
        
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
                
                # Calculate stops based on ATR
                atr = calculate_atr(
                    multi_tf_data["1h"]["highs"], 
                    multi_tf_data["1h"]["lows"], 
                    multi_tf_data["1h"]["closes"]
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
                    "strategy": "breakout"
                }
                
        elif regime == "volatile":
            # Mean reversion + momentum hybrid for volatile markets
            mean_rev_signal, mean_rev_reason = calculate_mean_reversion_signal(multi_tf_data, current_price)
            momentum_signal, momentum_reason = calculate_momentum_signal(multi_tf_data)
            
            # Combine signals (momentum gets 60% weight, mean reversion 40%)
            combined_signal = (momentum_signal * 0.6) + (mean_rev_signal * 0.4)
            
            if abs(combined_signal) > 0.6:  # Strong signal threshold
                direction = "LONG" if combined_signal > 0 else "SHORT"
                
                # Calculate stops
                atr = calculate_atr(
                    multi_tf_data["1h"]["highs"], 
                    multi_tf_data["1h"]["lows"], 
                    multi_tf_data["1h"]["closes"]
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
                    "signal_strength": abs(combined_signal)
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

def close_futures_position(symbol):
    try:
        positions = get_open_positions()
        if symbol not in positions: 
            print(f"No open position for {symbol}")
            return None
        pos = positions[symbol]
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

def execute_trade(coin, trade_params, current_price, balance):
    """Execute trade with fixed 15% position size and 15x leverage"""
    if trade_params["direction"] in ["SKIP", "GRID"]:
        return False
        
    try:
        symbol = f"{coin}USDT"
        
        # Fixed position sizing: 15% of portfolio at 15x leverage
        position_value = balance * POSITION_SIZE
        notional = position_value * LEVERAGE
        qty = q_qty(symbol, notional / current_price)
        
        if qty <= 0:
            print(f"Qty rounded to zero for {coin}")
            return False

        # Minimum notional check
        min_notional = _symbol_info.get(symbol, {}).get("minNotional", Decimal("5"))
        computed_notional = Decimal(str(current_price)) * Decimal(str(qty))
        if computed_notional < min_notional:
            print(f"Below MIN_NOTIONAL: {float(computed_notional):.4f} < {float(min_notional):.4f}")
            return False

        print(f"EXECUTING: {trade_params['direction']} {coin}")
        print(f"  Size: ${position_value:.0f} ({POSITION_SIZE*100}% of portfolio)")
        print(f"  Leverage: {LEVERAGE}x")
        print(f"  Notional: ${float(computed_notional):.0f}")
        print(f"  Strategy: {trade_params.get('strategy', 'unknown')}")
        print(f"  Reason: {trade_params['reason']}")

        # Execute the trade
        side = "BUY" if trade_params["direction"] == "LONG" else "SELL"
        res = place_futures_order(symbol, side, qty, LEVERAGE)
        
        if not res:
            print(f"Order failed for {symbol}")
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
                print("SL/TP failed, closing position")
                close_futures_position(symbol)
                return False

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

        print(f"SUCCESS: {coin} {trade_params['direction']} executed")
        return True

    except Exception as e:
        print(f"Trade execution error: {e}")
        return False

def handle_grid_trading(coin, symbol, current_price, balance):
    """Handle grid trading execution"""
    try:
        # Check if grid already exists
        existing_grid = db_execute("""
            SELECT COUNT(*) FROM grid_positions WHERE symbol = ? AND filled = FALSE
        """, (symbol,), fetch=True)
        
        if existing_grid[0][0] == 0:
            # Create new grid
            grid_id = setup_grid_trading(coin, symbol, current_price, balance)
            if not grid_id:
                return False
        
        # Check for triggered levels
        triggered_levels = check_grid_triggers(symbol, current_price)
        
        if triggered_levels:
            for level_id, level_price, level_type, grid_id in triggered_levels[:1]:  # Execute one at a time
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
                        
                        print(f"Grid {level_type} executed for {coin} at ${level_price:.4f}")
                        return True
        
        return False
        
    except Exception as e:
        print(f"Grid trading error: {e}")
        return False

# ============== Position Monitoring ==============
def monitor_positions():
    """Monitor positions for regime changes and profit taking"""
    try:
        positions = get_open_positions()
        
        for symbol, pos in positions.items():
            # Get fresh market data
            multi_tf_data = get_multi_timeframe_data(symbol)
            current_regime = detect_market_regime(multi_tf_data, pos["mark_price"])
            
            # Check for regime change exit
            should_close = False
            close_reason = ""
            
            # Get position details from database
            pos_details = db_execute("""
                SELECT strategy_type FROM active_positions WHERE symbol = ?
            """, (symbol,), fetch=True)
            
            if pos_details:
                strategy = pos_details[0][0]
                
                # Exit conditions based on strategy
                if strategy == "breakout":
                    # Exit breakout trades if regime changes to ranging/volatile
                    if current_regime in ["ranging", "volatile"]:
                        should_close = True
                        close_reason = f"Regime changed to {current_regime}"
                        
                elif strategy == "hybrid":
                    # Check if momentum has reversed
                    momentum_signal, _ = calculate_momentum_signal(multi_tf_data)
                    position_direction = 1 if pos["direction"] == "LONG" else -1
                    
                    if momentum_signal * position_direction < -0.3:  # Momentum reversed
                        should_close = True
                        close_reason = "Momentum reversal detected"
                
                # Profit taking for all strategies
                pnl_pct = (pos["pnl"] / (pos["notional"]/pos["leverage"])) * 100 if pos["notional"] > 0 else 0
                
                if pnl_pct > 8:  # Take profit at 8%
                    should_close = True
                    close_reason = f"Profit taking at {pnl_pct:.1f}%"
                elif pnl_pct < -4:  # Stop loss at -4% (backup to exchange SL)
                    should_close = True
                    close_reason = f"Emergency stop at {pnl_pct:.1f}%"
            
            if should_close:
                print(f"Closing {pos['coin']}: {close_reason}")
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
                    
                    print(f"Closed {pos['coin']}: P&L ${pnl_usd:+.2f} ({pnl_percent:+.1f}%)")
                    
        return positions
        
    except Exception as e:
        print(f"Monitor error: {e}")
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
        
        # Get recent trades
        recent_trades = db_execute("""
            SELECT timestamp, coin, action, direction, price, pnl, pnl_percent, reason
            FROM trades ORDER BY timestamp DESC LIMIT 20
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
        
        return jsonify({
            "portfolio_stats": stats,
            "positions": positions,
            "market_data": md,
            "trade_history": trade_history,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def run_flask_app():
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)

# ============== Main Trading Loop ==============
def run_technical_bot():
    """Main trading loop with technical analysis"""
    init_database()
    sync_server_time()
    load_exchange_info()

    print("Technical Analysis Trading Bot - Production Mode")
    print("Configuration:")
    print(f"  - Position Size: {POSITION_SIZE*100}% of portfolio")
    print(f"  - Leverage: {LEVERAGE}x")
    print(f"  - Max Positions: {MAX_CONCURRENT_POSITIONS}")
    print(f"  - Check Interval: {CHECK_INTERVAL}s")
    
    balance = get_futures_balance()
    if balance <= 0:
        print("No futures balance or API error")
        return
        
    print(f"Connected | Balance: ${balance:.2f}")

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
                print(f"\n=== Market Scan #{iteration} ===")
                
                balance = get_futures_balance()
                md = get_batch_market_data()
                stats = calculate_portfolio_stats(md)
                
                print(f"Portfolio: ${stats['total_value']:.2f} | ")
                print(f"Positions: {stats['positions_count']}/{MAX_CONCURRENT_POSITIONS} | ")
                print(f"Daily P&L: ${stats['daily_pnl']:+.2f}")
                
                # Identify high volume candidates
                high_volume_coins = identify_high_volume_coins(md)
                print(f"High volume coins: {len(high_volume_coins)} candidates")
                
                # Check if we can open new positions
                if stats['positions_count'] < MAX_CONCURRENT_POSITIONS:
                    max_new = MAX_CONCURRENT_POSITIONS - stats['positions_count']
                    executed = 0
                    
                    for coin, volume_score, coin_data in high_volume_coins[:15]:
                        if executed >= max_new:
                            break
                            
                        symbol = f"{coin}USDT"
                        if symbol in positions:
                            continue
                            
                        # Get technical analysis
                        multi_tf_data = get_multi_timeframe_data(symbol)
                        trade_decision = get_trade_decision(coin, md, multi_tf_data)
                        
                        if trade_decision["direction"] == "GRID":
                            # Handle grid trading
                            if handle_grid_trading(coin, symbol, coin_data["price"], balance):
                                executed += 1
                                print(f"Grid setup: {coin}")
                                
                        elif trade_decision["direction"] in ["LONG", "SHORT"]:
                            # Handle directional trades
                            if execute_trade(coin, trade_decision, coin_data["price"], balance):
                                executed += 1
                                time.sleep(2)  # Rate limiting
                    
                    if executed == 0:
                        print("No trading opportunities found")
                    else:
                        print(f"Executed {executed} new position(s)")
                        
                else:
                    print("Maximum positions reached")
                
                last_scan = now
                
            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            print("\nBot stopped by user")
            break
        except Exception as e:
            print(f"Main loop error: {e}")
            time.sleep(CHECK_INTERVAL)

# ============== Main Execution ==============
if __name__ == "__main__":
    # Start Flask dashboard
    flask_thread = threading.Thread(target=run_flask_app, daemon=True)
    flask_thread.start()
    
    # Start the technical trading bot
    run_technical_bot()