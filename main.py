# -*- coding: utf-8 -*-
import os, time, json, hmac, hashlib, math, threading, sqlite3, urllib.parse
from datetime import datetime, timedelta
from collections import deque
from decimal import Decimal, ROUND_DOWN

import requests
import numpy as np
from flask import Flask, jsonify, send_from_directory
import random

# ============== OpenAI (opsiyonel) ==============
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if (OpenAI and api_key) else None

# ============== Binance API ==============
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
    raise ValueError("Please set BINANCE_API_KEY and BINANCE_SECRET_KEY in your environment")

FAPI = "https://fapi.binance.com"  # Futures BASE (hem veri hem emir)
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

    # Basit backoff + zaman sync
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

# ============== Exchange Info Cache (precision/minNotional) ==============
_symbol_info = {}

def load_exchange_info():
    data = binance_request("/fapi/v1/exchangeInfo", "GET", signed=False)
    for s in data["symbols"]:
        sym = s["symbol"]
        pf = next(f for f in s["filters"] if f["filterType"] == "PRICE_FILTER")
        # MARKET_LOT_SIZE bazı sembollerde var; yoksa LOT_SIZE
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

# ============== Portföy & sabitler ==============
QUICK_CHECK_INTERVAL = 15
MAX_CONCURRENT_POSITIONS = 8
MIN_CONFIDENCE_THRESHOLD = 6

POSITION_SCALING = {
    0:0.20, 1:0.17, 2:0.14, 3:0.12,
    4:0.10, 5:0.08, 6:0.07, 7:0.06
}

FACTOR_WEIGHTS = {
    "timeframe_alignment": 0.25,
    "market_regime": 0.20,
    "momentum": 0.15,
    "volume_profile": 0.12,
    "technical_indicators": 0.10,
    "volatility": 0.08,
    "liquidity": 0.06,
    "divergences": 0.04
}

portfolio = {
    "balance": 0.0,
    "positions": {},
    "trade_history": [],
    "learning_data": {"successful_patterns":[], "failed_patterns":[], "performance_metrics":{}, "pattern_memory":{}}
}

# ============== DB (WAL + retry) ==============
DB_PATH = "trading_bot.db"

def db_connect():
    conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    return conn

def db_execute(sql, params=(), many=False, fetch=False):
    # Basit retry loop (database is locked)
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
    -- NEW: bandit stats per coin
    CREATE TABLE IF NOT EXISTS coin_stats (
        coin TEXT PRIMARY KEY,
        trades INTEGER DEFAULT 0,
        wins INTEGER DEFAULT 0,
        sum_pnl REAL DEFAULT 0,     -- sum of pnl_percent
        ema_wr REAL DEFAULT 0.5,    -- exponential moving win-rate
        ema_pnl REAL DEFAULT 0.0    -- exponential moving avg pnl_percent
    );
    """)
    conn.commit()
    conn.close()
    print("📁 SQLite initialized (WAL)")

# ============== Binance yardımcıları ==============
def get_futures_balance():
    try:
        info = binance_request("/fapi/v2/account", "GET", signed=True)
        if not info: return 0.0
        # totalWalletBalance USDT eşdeğeri
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
                "coin": coin,
                "symbol": sym,
                "size": amt,
                "direction": "LONG" if amt>0 else "SHORT",
                "entry_price": entry,
                "mark_price": mark,
                "pnl": pnl,
                "leverage": lev,
                "notional": abs(amt*mark)
            }
        return out
    except Exception as e:
        print(f"Positions error: {e}")
        return {}

def get_batch_market_data():
    """Futures 24hr tickers"""
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
    """Futures order book (notional-normalize, robust to 2-elem entries)."""
    try:
        data = binance_request("/fapi/v1/depth", "GET", params={"symbol": symbol, "limit": 100})
        raw_bids = data.get("bids", [])[:20]
        raw_asks = data.get("asks", [])[:20]

        # futures returns [price, qty] — but be defensive
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

        # simple normalization (tweak as you like)
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


# ============== Göstergeler / Analizler (mevcut mantığın iyileştirilmiş hali) ==============
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

def calculate_rsi(prices, period=14):
    if len(prices) < period+1: return 50
    gains, losses = [],[]
    for i in range(1,len(prices)):
        ch = prices[i]-prices[i-1]
        gains.append(max(ch,0)); losses.append(max(-ch,0))
    if len(gains)<period: return 50
    avg_gain = sum(gains[-period:])/period
    avg_loss = sum(losses[-period:])/period
    if avg_loss == 0: return 100
    rs = avg_gain/avg_loss
    return 100 - (100/(1+rs))

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    if len(prices) < period: return {"upper":0,"middle":0,"lower":0,"squeeze":False,"width":0}
    sma = sum(prices[-period:])/period
    var = sum((p-sma)**2 for p in prices[-period:]) / period
    std = var**0.5
    upper, lower = sma + std*std_dev, sma - std*std_dev
    width = (upper-lower)/sma if sma else 0
    return {"upper":upper, "middle":sma, "lower":lower, "squeeze": width<0.1, "width":width}

def calculate_ichimoku(highs,lows,closes):
    if len(highs)<52 or len(lows)<52 or len(closes)<52:
        return {"signal":"neutral","cloud_direction":"neutral"}
    tenkan = (max(highs[-9:])+min(lows[-9:]))/2
    kijun  = (max(highs[-26:])+min(lows[-26:]))/2
    senkou_a = (tenkan+kijun)/2
    senkou_b = (max(highs[-52:])+min(lows[-52:]))/2
    price = closes[-1]
    if price>max(senkou_a,senkou_b) and tenkan>kijun: sig="bullish"
    elif price<min(senkou_a,senkou_b) and tenkan<kijun: sig="bearish"
    else: sig="neutral"
    cd = "bullish" if senkou_a>senkou_b else "bearish"
    return {"signal":sig,"cloud_direction":cd,"tenkan":tenkan,"kijun":kijun}

def calculate_macd_with_divergence(prices, fast=12, slow=26, signal=9):
    if len(prices) < slow+signal: return {"signal":"neutral","divergence":None,"histogram":0,"macd_line":0}
    def ema(data,period):
        mult = 2/(period+1)
        e = sum(data[:period])/period
        for px in data[period:]:
            e = (px-e)*mult + e
        return e
    ema_fast = ema(prices,fast)
    ema_slow = ema(prices,slow)
    macd_line = ema_fast - ema_slow
    macd_vals=[]
    for i in range(slow,len(prices)):
        sub=prices[:i+1]
        if len(sub)>=slow:
            macd_vals.append(ema(sub,fast)-ema(sub,slow))
    if len(macd_vals) < signal: return {"signal":"neutral","divergence":None,"histogram":0,"macd_line":macd_line}
    sig_line = ema(macd_vals, signal)
    hist = macd_line - sig_line
    if macd_line>sig_line and hist>0: macd_sig="bullish"
    elif macd_line<sig_line and hist<0: macd_sig="bearish"
    else: macd_sig="neutral"
    div=None
    if len(prices)>=10 and len(macd_vals)>=10:
        rph, oph = max(prices[-5:]), max(prices[-10:-5])
        rmh, omh = max(macd_vals[-5:]), max(macd_vals[-10:-5])
        if rph>oph and rmh<omh: div="bearish"
        elif rph<oph and rmh>omh: div="bullish"
    return {"signal":macd_sig,"divergence":div,"histogram":hist,"macd_line":macd_line}

def analyze_momentum_volatility(multi_tf_data):
    if "1h" not in multi_tf_data or "closes" not in multi_tf_data["1h"]:
        return {"momentum_score":0,"volatility_cluster":False,"momentum_direction":"neutral","volatility_ratio":1}
    closes = multi_tf_data["1h"]["closes"]
    if len(closes)<20: return {"momentum_score":0,"volatility_cluster":False,"momentum_direction":"neutral","volatility_ratio":1}
    rets = [(closes[i]-closes[i-1])/closes[i-1] for i in range(1,len(closes))]
    recent = rets[-5:]; older = rets[-10:-5] if len(rets)>=10 else rets[:5]
    mom_score = (sum(recent)/len(recent)) - (sum(older)/len(older) if older else 0)
    direction = "bullish" if mom_score>0 else "bearish"
    vols = [abs(r) for r in rets[-20:]]
    recent_vol = sum(vols[-5:])/5
    avg_vol = sum(vols)/len(vols)
    v_cluster = recent_vol > avg_vol*1.5 if avg_vol>0 else False
    mom_norm = min(100, max(0, (mom_score + 0.02) * 2500))
    return {"momentum_score":mom_norm,"volatility_cluster":v_cluster,"momentum_direction":direction,"volatility_ratio": (recent_vol/avg_vol if avg_vol>0 else 1)}

def detect_market_regime(multi_tf_data, current_price):
    if not multi_tf_data or "1h" not in multi_tf_data:
        return {"regime":"UNKNOWN","confidence":0,"indicators":{}}
    hourly = multi_tf_data["1h"]
    closes = hourly.get("closes",[])
    highs  = hourly.get("highs",[])
    lows   = hourly.get("lows",[])
    if len(closes)<20: return {"regime":"UNKNOWN","confidence":0,"indicators":{}}

    def adx_simplified(highs,lows,closes,period=14):
        if len(highs)<period+1: return 0
        plus_dm, minus_dm, tr_vals = [],[],[]
        for i in range(1,len(highs)):
            hd = highs[i]-highs[i-1]
            ld = lows[i-1]-lows[i]
            plus_dm.append(max(hd,0) if hd>ld else 0)
            minus_dm.append(max(ld,0) if ld>hd else 0)
            tr = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
            tr_vals.append(tr)
        if not tr_vals: return 0
        avg_tr = sum(tr_vals[-period:])/period
        if avg_tr==0: return 0
        plus_di = (sum(plus_dm[-period:])/period)/avg_tr*100
        minus_di= (sum(minus_dm[-period:])/period)/avg_tr*100
        if plus_di+minus_di==0: return 0
        dx = abs(plus_di-minus_di)/(plus_di+minus_di)*100
        return dx

    adx = adx_simplified(highs,lows,closes)
    sma20 = sum(closes[-20:])/20
    sma50 = sum(closes[-50:])/50 if len(closes)>=50 else sma20
    price_above20 = current_price > sma20
    price_above50 = current_price > sma50
    if adx>25:
        if price_above20 and price_above50:
            return {"regime":"TRENDING_UP","confidence":min(adx/40*100,90),"indicators":{"adx":adx}}
        elif (not price_above20) and (not price_above50):
            return {"regime":"TRENDING_DOWN","confidence":min(adx/40*100,90),"indicators":{"adx":adx}}
    if adx<20:
        return {"regime":"RANGING","confidence":70,"indicators":{"adx":adx}}
    return {"regime":"TRANSITIONAL","confidence":50,"indicators":{"adx":adx}}

def analyze_timeframe_alignment(multi_tf_data):
    if not multi_tf_data:
        return {"aligned":False,"score":0,"direction":"neutral","signals":{},"strength":0}
    alignment_score=0; signals={}; strengths=[]
    for tf, data in multi_tf_data.items():
        closes = data.get("closes",[])
        if len(closes)>20:
            sma5 = sum(closes[-5:])/5
            sma10= sum(closes[-10:])/10
            sma20= sum(closes[-20:])/20
            tf_score=0
            if sma5>sma10> sma20:
                tf_score += 2; signals[tf]="strong_bullish"
            elif sma5> sma10 and sma10> sma20*0.995:
                tf_score +=1; signals[tf]="bullish"
            elif sma5< sma10< sma20:
                tf_score -=2; signals[tf]="strong_bearish"
            elif sma5< sma10 and sma10< sma20*1.005:
                tf_score -=1; signals[tf]="bearish"
            else:
                signals[tf]="neutral"
            alignment_score += tf_score
            strengths.append(abs(tf_score))
    avg_strength = sum(strengths)/len(strengths) if strengths else 0
    if alignment_score>=4: direction="strong_bullish"; aligned=True
    elif alignment_score>=2: direction="bullish"; aligned=True
    elif alignment_score<=-4: direction="strong_bearish"; aligned=True
    elif alignment_score<=-2: direction="bearish"; aligned=True
    else: direction="mixed"; aligned=False
    return {"aligned":aligned,"direction":direction,"score":alignment_score,"signals":signals,"strength":avg_strength}

def analyze_volume_profile(multi_tf_data):
    if "1h" not in multi_tf_data or "volumes" not in multi_tf_data["1h"]:
        return {"poc":0,"volume_trend":"neutral","volume_spike":False,"volume_ratio":1}
    vols = multi_tf_data["1h"]["volumes"]
    closes= multi_tf_data["1h"]["closes"]
    if len(vols)<10 or len(closes)<10:
        return {"poc":0,"volume_trend":"neutral","volume_spike":False,"volume_ratio":1}
    total = sum(vols[-20:])
    if total==0: return {"poc":0,"volume_trend":"neutral","volume_spike":False,"volume_ratio":1}
    vwap = sum(closes[i]*vols[i] for i in range(-20,0))/total
    recent = sum(vols[-5:])/5
    older  = sum(vols[-15:-10])/5 if len(vols)>=15 else recent
    if recent > older*1.3: trend="increasing"
    elif recent < older*0.7: trend="decreasing"
    else: trend="stable"
    latest = vols[-1]
    avg20  = sum(vols[-20:])/20
    spike  = latest > avg20*2
    return {"poc":vwap,"volume_trend":trend,"volume_spike":spike,"volume_ratio": (recent/older if older>0 else 1)}

# ============== Skor & karar entegrasyonu ==============
def get_comprehensive_analysis(symbol, market_data, multi_tf_data):
    coin = symbol.replace("USDT","")
    current_price = market_data[coin]["price"]
    analysis = {"overall_score":0,"direction":"neutral","confidence":0,"factors":{}}
    try:
        tf = analyze_timeframe_alignment(multi_tf_data)
        tf_score = 90 if (tf["aligned"] and tf["direction"].startswith("strong_")) else 70 if (tf["aligned"]) else 30
        analysis["factors"]["timeframe_alignment"] = {"score":tf_score,"weight":FACTOR_WEIGHTS["timeframe_alignment"],"direction":tf["direction"],"details":tf}

        regime = detect_market_regime(multi_tf_data, current_price)
        analysis["factors"]["market_regime"] = {"score":regime["confidence"],"weight":FACTOR_WEIGHTS["market_regime"],"regime":regime["regime"],"details":regime}

        mom = analyze_momentum_volatility(multi_tf_data)
        analysis["factors"]["momentum"] = {"score":mom["momentum_score"],"weight":FACTOR_WEIGHTS["momentum"],"direction":mom["momentum_direction"],"details":mom}

        volp = analyze_volume_profile(multi_tf_data)
        volp_score = 50 + (30 if volp["volume_trend"]=="increasing" else 0) + (20 if volp["volume_spike"] else 0)
        analysis["factors"]["volume_profile"] = {"score":volp_score,"weight":FACTOR_WEIGHTS["volume_profile"],"trend":volp["volume_trend"],"details":volp}

        tech_score = 50
        if "1h" in multi_tf_data and "closes" in multi_tf_data["1h"]:
            closes = multi_tf_data["1h"]["closes"]
            highs  = multi_tf_data["1h"]["highs"]
            lows   = multi_tf_data["1h"]["lows"]
            bb = calculate_bollinger_bands(closes)
            ich= calculate_ichimoku(highs,lows,closes)
            macd = calculate_macd_with_divergence(closes)
            if current_price > bb["upper"]: tech_score += 10
            elif current_price < bb["lower"]: tech_score -= 10
            if ich["signal"]=="bullish": tech_score += 15
            elif ich["signal"]=="bearish": tech_score -= 15
            if macd["signal"]=="bullish": tech_score += 10
            elif macd["signal"]=="bearish": tech_score -= 10
            analysis["factors"]["technical_indicators"] = {
                "score": max(0,min(100,tech_score)),
                "weight": FACTOR_WEIGHTS["technical_indicators"],
                "details":{"bollinger":bb,"ichimoku":ich,"macd":macd}
            }
        else:
            analysis["factors"]["technical_indicators"] = {"score":50,"weight":FACTOR_WEIGHTS["technical_indicators"],"details":None}

        vol_score = 50 + (20 if mom["volatility_cluster"] else 0)
        analysis["factors"]["volatility"] = {"score":vol_score,"weight":FACTOR_WEIGHTS["volatility"],"cluster":mom["volatility_cluster"],"details":mom}

        liq = get_order_book_analysis(symbol)
        analysis["factors"]["liquidity"] = {"score":liq["liquidity_score"],"weight":FACTOR_WEIGHTS["liquidity"],"imbalance":liq["imbalance_ratio"],"details":liq}

        total = 0
        for f,v in analysis["factors"].items():
            total += v["score"] * v["weight"]
        analysis["overall_score"] = total

        if total >= 65:
            analysis["direction"] = "bullish"; analysis["confidence"] = min(10, int((total-50)/5))
        elif total <= 35:
            analysis["direction"] = "bearish"; analysis["confidence"] = min(10, int((50-total)/5))
        else:
            analysis["direction"] = "neutral"; analysis["confidence"] = 3
        return analysis
    except Exception as e:
        print(f"Comprehensive analysis error: {e}")
        return analysis

# ============== SL/TP ve emirleme ==============
def calculate_dynamic_stops(current_price, atr, direction, market_regime):
    def get_hp(k, default): 
        rows = db_execute("SELECT value FROM hyperparameters WHERE key=?", (k,), fetch=True)
        return rows[0][0] if rows else default

    sl_mult = get_hp("sl_multiplier", 1.5)
    tp_mult = get_hp("tp_multiplier", 2.5)
    if market_regime in ("VOLATILE",):
        sl_mult, tp_mult = 2.0, 3.5
    elif market_regime=="RANGING":
        sl_mult, tp_mult = 1.0, 1.5
    elif market_regime in ("TRENDING_UP","TRENDING_DOWN"):
        sl_mult, tp_mult = 1.8, 4.0

    # subtle performance bias (better recent WR -> a bit tighter SL, fatter TP)
    gl = get_policy_adjustments()["global"]["win_rate"]  # 0..1 fraction
    bias = (gl - 0.50)  # -0.5..+0.5 -> typical -0.2..+0.2
    sl_mult *= (1.0 - 0.08 * bias)  # small tighten/loosen
    tp_mult *= (1.0 + 0.12 * bias)

    atr = atr or (current_price*0.005)  # fallback
    atr_sl = atr*sl_mult
    atr_tp = atr*tp_mult
    if direction=="LONG":
        sl = current_price - atr_sl
        tp = current_price + atr_tp
    else:
        sl = current_price + atr_sl
        tp = current_price - atr_tp

    sl_pct = abs(sl-current_price)/current_price
    tp_pct = abs(tp-current_price)/current_price

    min_sl,max_sl = 0.01,0.05
    min_tp,max_tp = 0.015,0.10

    if sl_pct < min_sl:
        sl = current_price*(1-min_sl if direction=="LONG" else 1+min_sl)
        sl_pct = min_sl
    elif sl_pct > max_sl:
        sl = current_price*(1-max_sl if direction=="LONG" else 1+max_sl)
        sl_pct = max_sl

    if tp_pct < min_tp:
        tp = current_price*(1+min_tp if direction=="LONG" else 1-min_tp)
        tp_pct = min_tp
    elif tp_pct > max_tp:
        tp = current_price*(1+max_tp if direction=="LONG" else 1-max_tp)
        tp_pct = max_tp

    rr = (tp_pct/sl_pct) if sl_pct>0 else 0
    return {"stop_loss":sl,"take_profit":tp,"sl_percentage":sl_pct*100,"tp_percentage":tp_pct*100,"risk_reward":rr}

def place_futures_order(symbol, side, quantity, leverage=10, order_type="MARKET"):
    try:
        # leverage
        _ = binance_request("/fapi/v1/leverage","POST",params={"symbol":symbol,"leverage":leverage},signed=True)
        # newClientOrderId idempotency
        cid = f"bot-{int(time.time()*1000)}-{side}"
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
        cid = f"bot-close-{int(time.time()*1000)}"
        res = binance_request("/fapi/v1/order","POST",signed=True,params={
            "symbol":symbol,"side":side,"type":"MARKET","quantity":q_qty(symbol,size),"reduceOnly":True,"newClientOrderId":cid
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
            "symbol": symbol,
            "side": sl_side,
            "type": "STOP_MARKET",
            "quantity": qty,
            "stopPrice": q_price(symbol, stop_price),
            "workingType": "MARK_PRICE",
            "reduceOnly": True
        })
        tp_order = binance_request("/fapi/v1/order","POST",signed=True,params={
            "symbol": symbol,
            "side": tp_side,
            "type": "TAKE_PROFIT_MARKET",
            "quantity": qty,
            "stopPrice": q_price(symbol, take_profit_price),
            "workingType": "MARK_PRICE",
            "reduceOnly": True
        })
        return {"stop_loss": sl_order, "take_profit": tp_order}
    except Exception as e:
        print(f"SL/TP error: {e}")
        return None


# ============== AI hafıza / maliyet ==============
cost_tracker = {'openai_tokens':0,'openai_cost':0.0,'api_calls_count':0,'railway_start_time':datetime.now(),'daily_costs':[],'last_reset':datetime.now()}

def get_ai_context_memory():
    rows = db_execute("""
        SELECT coin, direction, confidence, reasoning, outcome_pnl, timestamp, market_regime
        FROM ai_decisions ORDER BY timestamp DESC LIMIT 8
    """, fetch=True) or []
    if not rows: return "No previous AI decisions available. This is a fresh start."
    ctx = "RECENT AI DECISIONS & OUTCOMES:\n"
    wins=0; total=0
    for coin, direction, conf, reason, pnl, ts, market in rows:
        if pnl is not None:
            total += 1
            if pnl>0: wins += 1; outcome = f"+{pnl:.1f}% ✅"
            else: outcome = f"{pnl:.1f}% ❌"
        else: outcome = "OPEN 🔄"
        ctx += f"• {coin} {direction} (Conf:{conf}) → {outcome}\n"
        if reason: ctx += f"  Logic: {reason[:40]}...\n"
    if total>0:
        wr = wins/total*100
        ctx += f"\nRECENT PERFORMANCE: {wr:.0f}% win rate ({wins}/{total})\n"
        ctx += ("⚠️ Consider being more selective with confidence scores\n" if wr<50 else "✅ Good performance, maintain similar approach\n")
    ctx += "\nKEY: Learn from failures, build on successes, adjust confidence accordingly.\n"
    return ctx

def save_ai_decision(coin, trade_params):
    db_execute("""
        INSERT INTO ai_decisions (coin,direction,confidence,reasoning,market_regime,timestamp)
        VALUES (?,?,?,?,?,?)
    """, (coin, trade_params["direction"], trade_params["confidence"], (trade_params.get("reasoning") or "")[:200],
          trade_params.get("market_regime","UNKNOWN"), datetime.now().isoformat()))

def update_ai_decision_outcome(coin, direction, pnl_percent):
    db_execute("""
        UPDATE ai_decisions SET outcome_pnl=?
        WHERE coin=? AND direction=? AND outcome_pnl IS NULL
        ORDER BY timestamp DESC LIMIT 1
    """, (pnl_percent, coin, direction))

def track_cost(service, cost, units):
    db_execute("INSERT INTO cost_tracking (service,cost,units) VALUES (?,?,?)", (service, cost, units))
    if service=="openai":
        today = datetime.now().date()
        db_execute("""
            INSERT INTO daily_cost_summary (date, openai_cost, api_calls)
            VALUES (?, COALESCE((SELECT openai_cost FROM daily_cost_summary WHERE date=?),0)+?, 
                        COALESCE((SELECT api_calls FROM daily_cost_summary WHERE date=?),0)+1)
            ON CONFLICT(date) DO UPDATE SET
              openai_cost=excluded.openai_cost, api_calls=excluded.api_calls
        """, (today,today,cost,today))
        cost_tracker["openai_cost"] += cost
        cost_tracker["api_calls_count"] += 1

def get_cost_projections():
    rows = db_execute("""
        SELECT AVG(openai_cost), AVG(api_calls) FROM daily_cost_summary
        WHERE date >= date('now','-7 days')
    """, fetch=True)
    avg_openai = rows[0][0] if rows and rows[0] else None
    hours = (datetime.now()-cost_tracker["railway_start_time"]).total_seconds()/3600
    railway_hourly = 0.01
    current_railway = hours*railway_hourly
    if avg_openai is not None:
        daily_openai = avg_openai; weekly_openai = daily_openai*7; monthly_openai = daily_openai*30
    else:
        if hours>0:
            daily_openai = (cost_tracker["openai_cost"]/hours)*24
            weekly_openai = daily_openai*7; monthly_openai = daily_openai*30
        else:
            daily_openai=weekly_openai=monthly_openai=0
    weekly_rail = railway_hourly*24*7; monthly_rail = railway_hourly*24*30
    return {
        "current":{"openai":cost_tracker["openai_cost"],"railway":current_railway,
                   "total":cost_tracker["openai_cost"]+current_railway,"api_calls":cost_tracker["api_calls_count"]},
        "projections":{"weekly":{"openai":weekly_openai,"railway":weekly_rail,"total":weekly_openai+weekly_rail},
                       "monthly":{"openai":monthly_openai,"railway":monthly_rail,"total":monthly_openai+monthly_rail}}
    }


def get_hp_value(key, default):
    row = db_execute("SELECT value FROM hyperparameters WHERE key=?", (key,), fetch=True)
    try:
        return float(row[0][0]) if row and row[0] and row[0][0] is not None else default
    except:
        return default

def get_realized_pnl_today():
    rows = db_execute("""
        SELECT COALESCE(SUM(pnl), 0)
        FROM trades
        WHERE action LIKE '%CLOSE%' AND DATE(timestamp) = DATE('now')
    """, fetch=True)
    return float(rows[0][0] if rows and rows[0] and rows[0][0] is not None else 0.0)

def update_coin_stats(coin, pnl_percent):
    # EMA smoothing (keeps the brain “fresh”)
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
    """
    coin_scores: list of tuples (coin, score, ana, tf)
    Returns same list sorted by UCB score (desc), tie-break by |score-50|.
    """
    # Total trades to scale exploration
    trow = db_execute("SELECT COALESCE(SUM(trades),0) FROM coin_stats", fetch=True)
    total_trades = float(trow[0][0] if trow and trow[0] else 0.0) + 1.0

    out = []
    for coin, score, ana, tf in coin_scores:
        row = db_execute("SELECT trades, ema_pnl FROM coin_stats WHERE coin=?", (coin,), fetch=True)
        if row:
            n_i, ema_p = float(row[0][0] or 0.0), float(row[0][1] or 0.0)
        else:
            n_i, ema_p = 0.0, 0.0
        # UCB on pnl% (scaled). More trades → smaller explore term.
        explore = c * math.sqrt(math.log(total_trades) / max(1.0, n_i))
        ucb = ema_p + explore
        out.append((ucb, coin, score, ana, tf))
    out.sort(key=lambda x: (x[0], abs(x[2] - 50)), reverse=True)
    return [(coin, score, ana, tf) for (_ucb, coin, score, ana, tf) in out]

# ============== Öğrenme / risk ==============
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
    """
    Builds tiny, safe adjustments from recent outcomes:
      - global thresholds (bullish/bearish) move ±3 max
      - per-coin score bias ±3 max
      - per-coin size/leverage multipliers within tight bounds
    """
    rows = db_execute("""
        SELECT coin, outcome_pnl
        FROM ai_decisions
        WHERE outcome_pnl IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT 500
    """, fetch=True) or []

    # Global stats
    total = len(rows)
    wins = sum(1 for _, pnl in rows if (pnl or 0) > 0)
    global_wr = (wins / total) if total else 0.5

    # Dynamic thresholds (stay aggressive: bigger downshift when doing well; tiny upshift when doing poorly)
    # Base: 65/35  -> Adjust: (-3..+2)
    shift = clamp((global_wr - 0.50) * 10.0, -3.0, 2.0)
    bullish_thr = 65.0 - shift      # higher WR => slightly lower gate
    bearish_thr = 35.0 + shift      # higher WR => slightly lower (more aggressive) short gate too

    # Per-coin aggregates
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

        # Score bias: map win-rate to [-3, +3]
        bias = clamp((wr - 0.5) * 20.0, -3.0, 3.0)

        # Size & leverage multipliers (tight bounds)
        size_mult = clamp(1.0 + (wr - 0.5) * 0.30, 0.85, 1.15)
        lev_mult  = clamp(1.0 + (avg_pnl / 100.0) * 0.50, 0.90, 1.20)

        per_coin[coin] = {
            "bias": bias,
            "size_mult": size_mult,
            "lev_mult": lev_mult
        }

    # Exploration rate (default 10% unless overridden in DB)
    row = db_execute("SELECT value FROM hyperparameters WHERE key='exploration_rate'", fetch=True)
    exploration_rate = float(row[0][0]) if row and row[0] and row[0][0] is not None else 0.10
    exploration_rate = clamp(exploration_rate, 0.00, 0.30)

    return {
        "global": {
            "win_rate": global_wr,
            "bullish_thr": bullish_thr,
            "bearish_thr": bearish_thr,
            "exploration_rate": exploration_rate
        },
        "per_coin": per_coin
    }

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

        # Directional exposure
        long_notional  = sum(p["notional"] for p in positions.values() if p["direction"]=="LONG")
        short_notional = sum(p["notional"] for p in positions.values() if p["direction"]=="SHORT")
        max_dir_x = get_hp_value("max_directional_exposure_x", 6.0)  # e.g., 6x of balance per direction
        can_long  = (long_notional / max(1e-9, balance))  < max_dir_x
        can_short = (short_notional / max(1e-9, balance)) < max_dir_x

        # Daily loss breaker
        max_dd_pct = get_hp_value("max_daily_loss_pct", 8.0)  # ~aggressive but safe-ish
        realized_today = get_realized_pnl_today()
        dd_hit = (realized_today <= -abs(max_dd_pct) * balance / 100.0)

        out = {"positions_count":len(positions),
               "total_notional":total_notional,"total_margin_used":total_margin,
               "balance":balance,"margin_ratio":margin_ratio,"available_balance":available,
               "max_positions":MAX_CONCURRENT_POSITIONS,
               "can_trade":True,"reason":"Good to trade",
               "can_trade_long":can_long and not dd_hit,
               "can_trade_short":can_short and not dd_hit,
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

# ============== AI karar ==============
def ai_trade_decision(coin, market_data, multi_tf_data, learning_insights):
    """
    Deterministic trade planner:
      - Uses get_comprehensive_analysis and calculate_dynamic_stops
      - Converts score -> direction, confidence, leverage, base size
      - Enforces RR >= 1.7
      - Returns the same shape your pipeline expects
    """
    try:
        current_price = market_data[coin]["price"]
        symbol = f"{coin}USDT"

        # 1) Core analysis
        ana = get_comprehensive_analysis(symbol, market_data, multi_tf_data)
        score = float(ana.get("overall_score", 50.0))
        dir_hint = ana.get("direction", "neutral")

        # Map score → direction
        if score >= 65:
            direction = "LONG"
        elif score <= 35:
            direction = "SHORT"
        else:
            return {"direction": "SKIP", "reason": "Market conditions too neutral"}

        # 2) ATR & regime-aware stops
        hourly = multi_tf_data.get("1h", {})
        atr = 0.0
        if all(k in hourly for k in ("highs", "lows", "closes")):
            atr = calculate_atr(hourly["highs"], hourly["lows"], hourly["closes"])

        mreg = ana["factors"]["market_regime"]["details"]
        dyn = calculate_dynamic_stops(
            current_price,
            atr,
            "LONG" if direction == "LONG" else "SHORT",
            mreg.get("regime", "UNKNOWN")
        )

        # 3) Quick sanity checks (skip illiquid/odd scenarios)
        liq_score = ana["factors"].get("liquidity", {}).get("score", 50)
        if liq_score < 5:
            return {"direction": "SKIP", "reason": "Liquidity extremely low"}

        # 4) Require decent risk/reward
        rr = float(dyn["tp_percentage"]) / max(1e-6, float(dyn["sl_percentage"]))
        if rr < 1.7:
            return {"direction": "SKIP", "reason": f"RR {rr:.1f} < 1.7"}

        # 5) Confidence from score distance to 50
        #    65→~6, 75→~8, 85→~10  (mirrors for shorts)
        dist = abs(score - 50.0)
        # base: start at 6 when just over the gate, ramp to 10 at 85+
        confidence = 6 + int(max(0.0, (dist - 15.0)) // 5.0)
        confidence = int(max(6, min(10, confidence)))

        # If multi-timeframe alignment is "strong_*", nudge confidence +1
        tf_details = ana["factors"].get("timeframe_alignment", {}).get("details", {})
        if isinstance(tf_details, dict) and str(tf_details.get("direction", "")).startswith("strong_"):
            confidence = min(10, confidence + 1)

        # 6) Regime-aware base leverage (final lev/size still refined in execute_real_trade)
        reg = mreg.get("regime", "TRANSITIONAL")
        base_lev = 15 + (confidence - 6) * 3  # 15..30
        reg_boost = {"TRENDING_UP": 1.10, "TRENDING_DOWN": 1.10, "RANGING": 0.96, "TRANSITIONAL": 1.00}.get(reg, 1.00)
        leverage = int(max(10, min(30, round(base_lev * reg_boost))))

        # 7) Base position size 5–15% from confidence (exact $ sized later)
        base_size = 0.05 + (confidence - 6) * 0.02   # 0.05..0.15
        base_size = float(max(0.05, min(0.15, base_size)))

        # 8) Reason string (short but informative)
        top = sorted(
            ana["factors"].items(),
            key=lambda kv: kv[1].get("score", 0) * kv[1].get("weight", 0),
            reverse=True
        )[:2]
        top_names = ", ".join([k for k, _ in top]) if top else "n/a"
        reasoning = (
            f"Score {score:.1f} → {direction} | Regime {reg} | "
            f"RR~1:{rr:.1f} | Top factors: {top_names}"
        )

        # 9) Package result
        return {
            "direction": direction,
            "leverage": leverage,
            "position_size": base_size,          # execute_real_trade will still call get_dynamic_position_size
            "confidence": confidence,
            "reasoning": reasoning,
            "stop_loss": float(dyn["stop_loss"]),
            "take_profit": float(dyn["take_profit"]),
            "sl_percentage": float(dyn["sl_percentage"]),  # in %
            "tp_percentage": float(dyn["tp_percentage"]),  # in %
            "market_regime": reg,
            "atr": float(atr),
            "overall_score": float(score),
            "analysis_direction": dir_hint,
        }

    except Exception as e:
        print(f"AI(plan) error: {e}")
        return {"direction": "SKIP", "reason": f"Planner exception: {e}"}


def execute_real_trade(coin, trade_params, current_price,
                       bullish_thr=65.0, bearish_thr=35.0,
                       size_mult=1.0, lev_mult=1.0, score_bias=0.0):
    if trade_params["direction"] == "SKIP":
        return False
    try:
        symbol = f"{coin}USDT"

        # Use biased score for gating
        base_score = trade_params.get("overall_score", 50.0)
        score = base_score + float(score_bias or 0.0)

        if trade_params["direction"] == "LONG" and score < bullish_thr:
            print(f"Reject LONG: score {score:.1f} < gate {bullish_thr:.1f}")
            return False
        if trade_params["direction"] == "SHORT" and score > bearish_thr:
            print(f"Reject SHORT: score {score:.1f} > gate {bearish_thr:.1f}")
            return False

        # Portfolio/risk (directional caps + daily breaker)
        prisk = calculate_portfolio_risk()
        if not prisk["can_trade"]:
            print(f"❌ Cannot trade: {prisk['reason']}")
            return False
        if trade_params["direction"] == "LONG" and not prisk.get("can_trade_long", True):
            print("❌ Directional cap: cannot open more LONG exposure right now")
            return False
        if trade_params["direction"] == "SHORT" and not prisk.get("can_trade_short", True):
            print("❌ Directional cap: cannot open more SHORT exposure right now")
            return False

        balance = prisk["balance"]
        current_positions = get_open_positions()

        # Sizing (apply multipliers)
        sizing = get_dynamic_position_size(len(current_positions), trade_params["confidence"], balance)
        position_value = sizing["position_value"] * float(size_mult or 1.0)

        # Leverage (apply multiplier)
        leverage = int(round(clamp(trade_params["leverage"] * float(lev_mult or 1.0), 10, 30)))

        # Qty / notional
        notional = position_value * leverage
        qty = q_qty(symbol, notional / current_price)
        if qty <= 0:
            print("❌ Qty rounded to zero")
            return False

        # minNotional check
        min_notional = _symbol_info.get(symbol, {}).get("minNotional", Decimal("5"))
        computed_notional = Decimal(str(current_price)) * Decimal(str(qty))
        if computed_notional < min_notional:
            print(f"❌ Below MIN_NOTIONAL: notional={float(computed_notional):.4f} < {float(min_notional):.4f}")
            return False

        print(f"📊 Opening #{len(current_positions)+1}: {trade_params['direction']} {coin} "
              f"| qty={qty} | price≈{current_price:.6f} | notional≈{float(computed_notional):.2f} "
              f"| lev={leverage}x | score={score:.1f} (base {base_score:.1f} + bias {score_bias:+.1f})")

        # Place market order
        side = "BUY" if trade_params["direction"] == "LONG" else "SELL"
        res = place_futures_order(symbol, side, qty, leverage)
        if not res:
            print(f"❌ Order failed for {symbol}")
            return False

        # Place SL/TP (reduceOnly, MARK_PRICE) + fail-safe
        sltp = set_stop_loss_take_profit(symbol, trade_params["stop_loss"], trade_params["take_profit"], trade_params["direction"])
        if not sltp or not sltp.get("stop_loss") or not sltp.get("take_profit"):
            print("⚠️ SL/TP placement failed, retrying once...")
            sltp = set_stop_loss_take_profit(symbol, trade_params["stop_loss"], trade_params["take_profit"], trade_params["direction"])
            if not sltp or not sltp.get("stop_loss") or not sltp.get("take_profit"):
                print("❌ SL/TP failed twice — closing immediately to avoid naked exposure.")
                close_futures_position(symbol)
                return False

        # Save AI decision & trade row (OPEN)
        save_ai_decision(coin, trade_params)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pid = f"{symbol}_{ts.replace(' ','_').replace(':','-')}"
        db_execute("""
            INSERT OR REPLACE INTO trades (timestamp,position_id,coin,action,direction,price,position_size,leverage,
            notional_value,stop_loss,take_profit,confidence,reason)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (ts, pid, coin, f"{trade_params['direction']} OPEN", trade_params["direction"], current_price,
              position_value, leverage, float(computed_notional), trade_params["stop_loss"],
              trade_params["take_profit"], trade_params["confidence"], trade_params.get("reasoning","")))

        print(f"✅ EXECUTED: {coin} {trade_params['direction']} {qty} @ {leverage}x")
        return True

    except Exception as e:
        print(f"Trade exec error: {e}")
        return False
_pyramid_counts = {}  # symbol -> int

def pyramid_winner(symbol, pos, multi_tf_data, md):
    """
    Add to a winning position up to 3 times if momentum persists.
    Each add ~30% of current notional, same leverage.
    """
    try:
        coin = pos["coin"]
        price = pos["mark_price"]; entry = pos["entry_price"]
        direction = pos["direction"]; lev = int(pos["leverage"])
        pnl_pct = (price-entry)/entry if direction=="LONG" else (entry-price)/entry

        # Conditions to consider add
        if pnl_pct < 0.03:   # at least +3%
            return False
        if "1h" not in multi_tf_data or "closes" not in multi_tf_data["1h"]:
            return False

        # Momentum still aligned with direction?
        ana = get_comprehensive_analysis(symbol, {coin: {"price": price, "change_24h":0, "volume":0}}, multi_tf_data)
        if direction=="LONG" and ana["direction"] not in ("bullish",):
            return False
        if direction=="SHORT" and ana["direction"] not in ("bearish",):
            return False

        # Limit adds
        n = _pyramid_counts.get(symbol, 0)
        if n >= 3:
            return False

        # Position add sizing: ~30% of current notional
        add_value = pos["notional"] * 0.30
        qty = q_qty(symbol, add_value / price)
        if qty <= 0:
            return False

        side = "BUY" if direction=="LONG" else "SELL"
        res = place_futures_order(symbol, side, qty, lev)
        if not res:
            print(f"❌ Pyramid add failed for {symbol}")
            return False

        # Refresh SL/TP to cover increased size (reuse last known SL/TP)
        set_stop_loss_take_profit(symbol, ana["factors"]["technical_indicators"]["details"]["bollinger"]["middle"] if ana["direction"]=="bullish" else price*1.02,
                                  price*1.06 if direction=="LONG" else price*0.94,
                                  direction)

        _pyramid_counts[symbol] = n + 1
        print(f"➕ Pyramided {coin}: add #{_pyramid_counts[symbol]} (qty={qty})")
        return True
    except Exception as e:
        print(f"Pyramid error: {e}")
        return False




# ============== Pozisyon izleme & çıkış ==============
def calculate_trailing_stop(position_data, multi_tf_data, atr):
    try:
        current_price = position_data["mark_price"]; entry_price = position_data["entry_price"]
        direction = position_data["direction"]
        pnl_pct = (current_price-entry_price)/entry_price if direction=="LONG" else (entry_price-current_price)/entry_price
        base = atr*1.5
        mult = 0.8 if pnl_pct>0.05 else 1.0 if pnl_pct>0.03 else 1.2
        dist = base*mult
        new_stop = current_price - dist if direction=="LONG" else current_price + dist
        return {"new_stop":new_stop,"trail_distance":dist,"should_update": pnl_pct>0.02}
    except Exception as e:
        print(f"Trailing error: {e}")
        return {"new_stop":0,"trail_distance":0,"should_update":False}

def calculate_partial_profit_levels(position_data, multi_tf_data):
    try:
        entry = position_data["entry_price"]; price = position_data["mark_price"]
        direction = position_data["direction"]
        pnl_pct = (price-entry)/entry if direction=="LONG" else (entry-price)/entry
        levels=[]
        if pnl_pct>=0.025: levels.append({"level":0.025,"percentage":25,"reason":"Initial profit taking"})
        if pnl_pct>=0.05:  levels.append({"level":0.05,"percentage":50,"reason":"Strong move"})
        if pnl_pct>=0.08:  levels.append({"level":0.08,"percentage":75,"reason":"Exceptional"})
        return levels
    except Exception:
        return []

def check_profit_taking_opportunity(position_data, multi_tf_data):
    price = position_data["mark_price"]; entry = position_data["entry_price"]
    direction = position_data["direction"]
    pnl_pct = (price-entry)/entry if direction=="LONG" else (entry-price)/entry
    if pnl_pct <= 0.015: return (False, None)

    symbol = position_data["symbol"]; coin = position_data["coin"]
    market_data = {coin: {"price": price, "change_24h":0, "volume":0}}
    analysis = get_comprehensive_analysis(symbol, market_data, multi_tf_data)
    signals=[]; strength=0

    if direction=="LONG" and analysis["direction"]=="bearish" and analysis["confidence"]>=6:
        signals.append("Market turned bearish"); strength+=3
    elif direction=="SHORT" and analysis["direction"]=="bullish" and analysis["confidence"]>=6:
        signals.append("Market turned bullish"); strength+=3

    if "1h" in multi_tf_data and "closes" in multi_tf_data["1h"]:
        closes = multi_tf_data["1h"]["closes"]
        rsi = calculate_rsi(closes)
        bb = calculate_bollinger_bands(closes)
        if direction=="LONG" and rsi>78: signals.append(f"RSI overbought {int(rsi)}"); strength+=2
        if direction=="SHORT" and rsi<22: signals.append(f"RSI oversold {int(rsi)}"); strength+=2
        if direction=="LONG" and price > bb["upper"]*1.02: signals.append("Above upper BB"); strength+=1
        if direction=="SHORT" and price < bb["lower"]*0.98: signals.append("Below lower BB"); strength+=1

    volp = analyze_volume_profile(multi_tf_data)
    if volp["volume_trend"]=="decreasing" and pnl_pct>0.03:
        signals.append("Volume declining"); strength+=1

    mom = analyze_momentum_volatility(multi_tf_data)
    if (direction=="LONG" and mom["momentum_direction"]=="bearish") or (direction=="SHORT" and mom["momentum_direction"]=="bullish"):
        signals.append("Momentum turning"); strength+=2

    partial = calculate_partial_profit_levels(position_data, multi_tf_data)
    if partial: signals.append(f"Reached target ({pnl_pct*100:.1f}%)"); strength+=1

    if strength>=4 and pnl_pct>0.02: return (True, f"Smart full_exit: {', '.join(signals)} (Profit {pnl_pct*100:.1f}%)")
    if strength>=2 and pnl_pct>0.04: return (True, f"Smart partial_exit: {', '.join(signals)} (Profit {pnl_pct*100:.1f}%)")
    if pnl_pct>0.08: return (True, f"Smart partial_exit: profit>8%")
    return (False, None)

def monitor_positions():
    try:
        positions = get_open_positions()
        for symbol, pos in positions.items():
            mtd = get_multi_timeframe_data(symbol)
            atr = 0
            if "1h" in mtd and all(k in mtd["1h"] for k in ("highs","lows","closes")):
                atr = calculate_atr(mtd["1h"]["highs"], mtd["1h"]["lows"], mtd["1h"]["closes"])

            # Try pyramiding winners (before exit checks)
            try:
                md = {pos["coin"]: {"price": pos["mark_price"], "change_24h":0, "volume":0}}
                pyramid_winner(symbol, pos, mtd, md)
            except Exception as _e:
                pass

            should_close, reason = check_profit_taking_opportunity(pos, mtd)
            if should_close:
                print(f"🧠 Exit {pos['coin']}: {reason}")
                res = close_futures_position(symbol)
                if res:
                    print(f"✅ Closed: {pos['coin']}")
                    # Approx realized PnL in $, and % on margin
                    realized_usd = float(pos["pnl"])
                    pnl_percent = (pos["pnl"] / (pos["notional"]/max(1,pos["leverage"]))) * 100 if pos["notional"]>0 else 0.0
                    update_ai_decision_outcome(pos["coin"], pos["direction"], pnl_percent)
                    update_coin_stats(pos["coin"], pnl_percent)

                    # Log CLOSE trade row
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    db_execute("""
                        INSERT INTO trades (timestamp, position_id, coin, action, direction, price,
                                            position_size, leverage, notional_value, stop_loss, take_profit,
                                            pnl, pnl_percent, reason, confidence, profitable)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (ts, f"{symbol}_CLOSE_{ts.replace(' ','_').replace(':','-')}",
                          pos["coin"], f"{pos['direction']} CLOSE", pos["direction"], pos["mark_price"],
                          None, pos["leverage"], pos["notional"], None, None,
                          realized_usd, pnl_percent, reason, None, 1 if pnl_percent>0 else 0))
                    # Reset pyramid counter for this symbol
                    _pyramid_counts.pop(symbol, None)
                else:
                    print(f"❌ Close failed: {pos['coin']}")
            else:
                if atr>0:
                    trail = calculate_trailing_stop(pos, mtd, atr)
                    if trail["should_update"]:
                        print(f"📈 Trailing stop opportunity {pos['coin']}: {trail['new_stop']:.4f}")
        return positions
    except Exception as e:
        print(f"Monitor error: {e}")
        return {}


def save_trade_to_db(trade_record):
    db_execute("""
        INSERT OR REPLACE INTO trades 
        (timestamp, position_id, coin, action, direction, price, position_size, leverage, notional_value, 
         stop_loss, take_profit, pnl, pnl_percent, duration, reason, confidence, profitable)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (trade_record.get("time"), trade_record.get("position_id"), trade_record.get("coin"),
          trade_record.get("action"), trade_record.get("direction"), trade_record.get("price"),
          trade_record.get("position_size"), trade_record.get("leverage"), trade_record.get("notional_value"),
          trade_record.get("stop_loss"), trade_record.get("take_profit"), trade_record.get("pnl"),
          trade_record.get("pnl_percent"), trade_record.get("duration"), trade_record.get("reason"),
          trade_record.get("confidence"), (trade_record.get("pnl",0)>0) if "pnl" in trade_record else None))

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
        return jsonify({
            "total_value": total_value,
            "balance": portfolio["balance"],
            "positions": portfolio["positions"],
            "trade_history": portfolio["trade_history"][-20:],
            "market_data": md,
            "learning_metrics": insights,
            "cost_tracking": cost_proj,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def run_flask_app():
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)

# ============== Ana döngü ==============
def run_enhanced_bot():
    init_database()
    sync_server_time()
    load_exchange_info()

    print("🚀 ENHANCED SMART TRADING BOT (Futures-only, precision-safe)")
    balance = get_futures_balance()
    if balance <= 0:
        print("❌ No futures balance or API error")
        return
    print(f"✅ Connected | Balance: ${balance:.2f}")

    target_coins = ['BTC','ETH','SOL','BNB','SEI','DOGE','TIA','TAO','ARB','SUI','ENA','FET']
    last_full = 0
    iteration = 0

    while True:
        try:
            iteration += 1
            now = time.time()
            current_positions = monitor_positions()

            if now - last_full >= 180:
                print("\n" + "="*80)
                print(f"🧠 ANALYSIS #{iteration} @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("="*80)

                current_balance = get_futures_balance()
                portfolio["balance"] = current_balance
                portfolio["positions"] = current_positions
                md = get_batch_market_data()

                print("\n💰 Account:")
                print(f"   Balance: ${current_balance:.2f}")
                print(f"   Open Positions: {len(current_positions)}")
                if current_positions:
                    print(f"   Unrealized P&L: ${sum(p['pnl'] for p in current_positions.values()):+.2f}")

                print("\n📊 Market Overview:")
                for coin in target_coins[:6]:
                    if coin in md:
                        price = md[coin]["price"]
                        sym = f"{coin}USDT"
                        tf = get_multi_timeframe_data(sym)
                        ana = get_comprehensive_analysis(sym, {coin: md[coin]}, tf)
                        regime = detect_market_regime(tf, price)["regime"]
                        trend = "🚀" if regime == "TRENDING_UP" else "📉" if regime == "TRENDING_DOWN" else "↔️" if regime == "RANGING" else "❓"
                        pos_info = ""
                        if sym in current_positions:
                            pos = current_positions[sym]
                            pos_info = f" [{pos['direction']} {pos['pnl']:+.1f}]"
                        score = ana["overall_score"]
                        emoji = "🟢" if score > 65 else "🔴" if score < 35 else "🟡"
                        print(f"   {coin}: ${price:.2f} {trend} ({md[coin]['change_24h']:+.1f}%) Score:{score:.0f} {emoji}{pos_info}")

                prisk = calculate_portfolio_risk()
                if prisk["can_trade"]:
                    insights = get_learning_insights()
                    print("\n🤖 Opportunity Scan:")
                    print(f"   Portfolio {prisk['positions_count']}/{MAX_CONCURRENT_POSITIONS} | Avail ${prisk['available_balance']:.2f} | Margin {prisk['margin_ratio']*100:.1f}% | Win {insights['win_rate']:.1f}%")

                    # --- Learning & policy adjustments ---
                    policy = get_policy_adjustments()
                    b_thr = policy["global"]["bullish_thr"]
                    s_thr = policy["global"]["bearish_thr"]
                    explore = (random.random() < policy["global"]["exploration_rate"])

                    coin_scores = []
                    for coin in target_coins:
                        if coin not in md:
                            continue
                        sym = f"{coin}USDT"
                        if sym in current_positions:
                            continue
                        tf = get_multi_timeframe_data(sym)
                        ana = get_comprehensive_analysis(sym, md, tf)
                        if ana["overall_score"] > 65 or ana["overall_score"] < 35:
                            coin_scores.append((coin, ana["overall_score"], ana, tf))

                    # UCB rank (monster smarts): exploration + exploitation
                    coin_scores = rank_candidates_ucb(coin_scores)

                    max_new = min(3, MAX_CONCURRENT_POSITIONS - prisk["positions_count"])
                    executed = 0

                    for i, (coin, score, ana, tf) in enumerate(coin_scores[:max_new]):
                        print(f"\n   📊 Candidate #{i+1}: {coin} Score {score:.1f} Dir {ana['direction']}")
                        tp = ai_trade_decision(coin, md, tf, insights)
                        if tp and tp["direction"] != "SKIP":
                            # sl/tp returned as percentages; default sl% to 1.0 to avoid division by zero
                            rr_disp = tp.get("tp_percentage", 0.0) / max(1e-6, tp.get("sl_percentage", 1.0))
                            print(f"      AI: {tp['direction']} conf {tp['confidence']}/10 rr≈1:{rr_disp:.1f}")

                            if tp["confidence"] >= MIN_CONFIDENCE_THRESHOLD:
                                rr = rr_disp
                                if rr >= 1.7:
                                    # Per-coin adjustments
                                    adj = policy["per_coin"].get(coin, {"bias": 0.0, "size_mult": 1.0, "lev_mult": 1.0})

                                    # Regime-aware aggression (bigger in trends)
                                    reg = ana["factors"]["market_regime"]["details"]["regime"]
                                    reg_size = {"TRENDING_UP":1.15, "TRENDING_DOWN":1.15, "RANGING":0.92, "TRANSITIONAL":1.0}.get(reg,1.0)
                                    reg_lev  = {"TRENDING_UP":1.10, "TRENDING_DOWN":1.10, "RANGING":0.96, "TRANSITIONAL":1.0}.get(reg,1.0)

                                    size_mult = adj["size_mult"] * reg_size
                                    lev_mult  = adj["lev_mult"]  * reg_lev

                                    # Slight gate relax for top candidate during exploration
                                    use_b_thr = b_thr - (2.0 if (explore and i == 0) else 0.0)
                                    use_s_thr = s_thr + (2.0 if (explore and i == 0) else 0.0)

                                    ok = execute_real_trade(
                                        coin, tp, md[coin]["price"],
                                        bullish_thr=use_b_thr,
                                        bearish_thr=use_s_thr,
                                        size_mult=size_mult,
                                        lev_mult=lev_mult,
                                        score_bias=adj["bias"]
                                    )
                                    if ok:
                                        executed += 1
                                        time.sleep(1)
                                    else:
                                        print("      ⚠️ Trade not executed (rejected or failed)")
                                else:
                                    print("      ⏭️ Low R/R")
                            else:
                                print("      ⏭️ Low confidence")
                        else:
                            print("      ⏭️ AI says SKIP")

                    if executed == 0:
                        print("   No suitable opportunities.")
                    else:
                        print(f"   🎯 Executed {executed} trade(s)")
                else:
                    print(f"\n⚠️ Cannot open new positions: {prisk['reason']}")

                total_value = calculate_portfolio_value(md)
                cost_data = get_cost_projections()
                final_risk = calculate_portfolio_risk()
                print("\n📈 Summary:")
                print(f"   Total Value: ${total_value:.2f}")
                print(f"   Active Positions: {final_risk['positions_count']}/{MAX_CONCURRENT_POSITIONS}")
                print(f"   Margin Usage: {final_risk['margin_ratio']*100:.1f}%  | Available: ${final_risk['available_balance']:.2f}")
                print(f"   Session Costs: ${cost_data['current']['total']:.4f}")
                last_full = now
                print("="*80)

            time.sleep(QUICK_CHECK_INTERVAL)

        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Loop error: {e}")
            time.sleep(QUICK_CHECK_INTERVAL)




# ============== Main ==============
if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask_app, daemon=True)
    flask_thread.start()
    run_enhanced_bot()
