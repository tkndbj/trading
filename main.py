import requests
import time
import os
from openai import OpenAI
import statistics
import json
import hmac
import hashlib
import urllib.parse
from datetime import datetime, timedelta
from flask import Flask, jsonify, send_from_directory
import threading
import math
import sqlite3
from collections import deque


api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

# Binance API credentials
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
BINANCE_BASE_URL = "https://fapi.binance.com"

if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
    raise ValueError("Please set BINANCE_API_KEY and BINANCE_SECRET_KEY in your .env file")

# Cost tracking with projections
cost_tracker = {
    'openai_tokens': 0,
    'openai_cost': 0.0,
    'api_calls_count': 0,
    'railway_start_time': datetime.now(),
    'daily_costs': [],
    'last_reset': datetime.now()
}

# Constants for optimization
QUICK_CHECK_INTERVAL = 15
FULL_ANALYSIS_INTERVAL = 60
MAX_CONCURRENT_POSITIONS = 4
MIN_CONFIDENCE_THRESHOLD = 6
BATCH_API_WEIGHT_LIMIT = 100

# Binance API helper functions
def get_binance_signature(query_string):
    """Generate signature for Binance API"""
    return hmac.new(
        BINANCE_SECRET_KEY.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

def binance_request(endpoint, method="GET", params=None, signed=False):
    """Make authenticated request to Binance API"""
    if params is None:
        params = {}
    
    headers = {
        'X-MBX-APIKEY': BINANCE_API_KEY
    }
    
    if signed:
        params['timestamp'] = int(time.time() * 1000)
        query_string = urllib.parse.urlencode(params)
        params['signature'] = get_binance_signature(query_string)
    
    url = f"{BINANCE_BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, params=params)
        elif method == "POST":
            response = requests.post(url, headers=headers, params=params)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, params=params)
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Binance API error: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        return None

def get_futures_balance():
    """Get futures account balance (USDC or BNFCR)"""
    try:
        account_info = binance_request("/fapi/v2/account", signed=True)
        if not account_info:
            return 0
        
        # Look for USDC or BNFCR balance
        for asset in account_info.get('assets', []):
            if asset['asset'] in ['USDC', 'BNFCR']:
                available_balance = float(asset['availableBalance'])
                if available_balance > 0:
                    print(f"💰 Futures Balance: {available_balance:.2f} {asset['asset']}")
                    return available_balance
        
        # Fallback to total wallet balance if no specific asset found
        total_balance = float(account_info.get('totalWalletBalance', 0))
        print(f"💰 Total Futures Balance: {total_balance:.2f} USDT")
        return total_balance
        
    except Exception as e:
        print(f"Error getting futures balance: {e}")
        return 0

def get_open_positions():
    """Get current open positions from Binance"""
    try:
        positions = binance_request("/fapi/v2/positionRisk", signed=True)
        if not positions:
            return {}
        
        open_positions = {}
        for pos in positions:
            size = float(pos['positionAmt'])
            if abs(size) > 0:  # Only positions with size > 0
                symbol = pos['symbol']
                coin = symbol.replace('USDT', '').replace('USDC', '').replace('BNFCR', '')
                
                # Handle different API response formats
                unrealized_pnl = pos.get('unRealizedPnl') or pos.get('unrealizedPnl') or '0'
                entry_price = pos.get('entryPrice') or pos.get('avgPrice') or '0'
                mark_price = pos.get('markPrice') or pos.get('lastPrice') or '0'
                
                open_positions[symbol] = {
                    'coin': coin,
                    'symbol': symbol,
                    'size': size,
                    'direction': 'LONG' if size > 0 else 'SHORT',
                    'entry_price': float(entry_price),
                    'mark_price': float(mark_price),
                    'pnl': float(unrealized_pnl),
                    'leverage': int(float(pos.get('leverage', '1'))),
                    'notional': abs(size * float(mark_price))
                }
        
        return open_positions
        
    except Exception as e:
        print(f"Error getting positions: {e}")
        import traceback
        traceback.print_exc()
        return {}

def place_futures_order(symbol, side, quantity, leverage=10, order_type="MARKET"):
    """Place futures order on Binance"""
    try:
        # Set leverage first
        leverage_result = binance_request(
            "/fapi/v1/leverage",
            method="POST",
            params={
                'symbol': symbol,
                'leverage': leverage
            },
            signed=True
        )
        
        if not leverage_result:
            print(f"Failed to set leverage for {symbol}")
            return None
        
        # Place order
        order_params = {
            'symbol': symbol,
            'side': side,  # BUY or SELL
            'type': order_type,
            'quantity': quantity
        }
        
        order_result = binance_request(
            "/fapi/v1/order",
            method="POST",
            params=order_params,
            signed=True
        )
        
        return order_result
        
    except Exception as e:
        print(f"Error placing order: {e}")
        return None

def close_futures_position(symbol):
    """Close futures position by placing opposite order"""
    try:
        # Get current position
        positions = get_open_positions()
        if symbol not in positions:
            print(f"No open position for {symbol}")
            return None
        
        position = positions[symbol]
        size = abs(position['size'])
        
        # Place opposite order to close
        side = "SELL" if position['direction'] == 'LONG' else "BUY"
        
        close_order = binance_request(
            "/fapi/v1/order",
            method="POST",
            params={
                'symbol': symbol,
                'side': side,
                'type': 'MARKET',
                'quantity': size,
                'reduceOnly': True
            },
            signed=True
        )
        
        return close_order
        
    except Exception as e:
        print(f"Error closing position: {e}")
        return None

def set_stop_loss_take_profit(symbol, stop_price, take_profit_price, position_side):
    """Set stop loss and take profit orders"""
    try:
        # Get position info
        positions = get_open_positions()
        if symbol not in positions:
            return None
        
        position = positions[symbol]
        quantity = abs(position['size'])
        
        # Stop Loss Order
        if position_side == "LONG":
            sl_side = "SELL"
            tp_side = "SELL"
        else:
            sl_side = "BUY"
            tp_side = "BUY"
        
        # Place Stop Loss
        sl_order = binance_request(
            "/fapi/v1/order",
            method="POST",
            params={
                'symbol': symbol,
                'side': sl_side,
                'type': 'STOP_MARKET',
                'quantity': quantity,
                'stopPrice': stop_price,
                'reduceOnly': True
            },
            signed=True
        )
        
        # Place Take Profit
        tp_order = binance_request(
            "/fapi/v1/order",
            method="POST",
            params={
                'symbol': symbol,
                'side': tp_side,
                'type': 'TAKE_PROFIT_MARKET',
                'quantity': quantity,
                'stopPrice': take_profit_price,
                'reduceOnly': True
            },
            signed=True
        )
        
        return {'stop_loss': sl_order, 'take_profit': tp_order}
        
    except Exception as e:
        print(f"Error setting SL/TP: {e}")
        return None

# Enhanced Database setup
def init_database():
    """Initialize SQLite database with enhanced tables"""
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    
    # Enhanced trades table
    cursor.execute('''
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
        )
    ''')
    
    # Active positions table for persistence
    cursor.execute('''
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
        )
    ''')
    
    # AI Decisions table for memory feature
    cursor.execute('''
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
        )
    ''')
    
    # Hyperparameters table for auto-tuning
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS hyperparameters (
            key TEXT PRIMARY KEY,
            value REAL NOT NULL
        )
    ''')
    
    # Cost tracking table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cost_tracking (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            service TEXT NOT NULL,
            cost REAL NOT NULL,
            units TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Daily cost summary for projections
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_cost_summary (
            date DATE PRIMARY KEY,
            openai_cost REAL DEFAULT 0,
            api_calls INTEGER DEFAULT 0,
            trades_executed INTEGER DEFAULT 0,
            positions_checked INTEGER DEFAULT 0
        )
    ''')
    
    conn.commit()
    conn.close()
    print("📁 Enhanced SQLite database initialized with AI memory")

# ========== FEATURE 1: AI MEMORY SYSTEM ==========
def get_ai_context_memory():
    """Get recent AI decisions for context continuity"""
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT coin, direction, confidence, reasoning, outcome_pnl, 
               timestamp, market_regime
        FROM ai_decisions 
        ORDER BY timestamp DESC 
        LIMIT 8
    ''')
    
    recent_decisions = cursor.fetchall()
    conn.close()
    
    if not recent_decisions:
        return "No previous AI decisions available. This is a fresh start."
    
    context = "RECENT AI DECISIONS & OUTCOMES:\n"
    winning_decisions = 0
    total_decisions = 0
    
    for decision in recent_decisions:
        coin, direction, confidence, reason, pnl, timestamp, market = decision
        if pnl is not None:
            total_decisions += 1
            if pnl > 0:
                winning_decisions += 1
                outcome = f"+{pnl:.1f}% ✅"
            else:
                outcome = f"{pnl:.1f}% ❌"
        else:
            outcome = "OPEN 🔄"
        
        context += f"• {coin} {direction} (Conf:{confidence}) → {outcome}\n"
        if reason:
            context += f"  Logic: {reason[:40]}...\n"
    
    if total_decisions > 0:
        win_rate = (winning_decisions / total_decisions) * 100
        context += f"\nRECENT PERFORMANCE: {win_rate:.0f}% win rate ({winning_decisions}/{total_decisions})\n"
        
        if win_rate < 50:
            context += "⚠️ Consider being more selective with confidence scores\n"
        elif win_rate > 70:
            context += "✅ Good performance, maintain similar approach\n"
    
    context += "\nKEY: Learn from failures, build on successes, adjust confidence accordingly.\n"
    
    return context

def save_ai_decision(coin, trade_params):
    """Save AI decision for future context"""
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO ai_decisions 
        (coin, direction, confidence, reasoning, market_regime, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        coin,
        trade_params['direction'],
        trade_params['confidence'],
        trade_params['reasoning'][:200] if trade_params.get('reasoning') else '',
        trade_params.get('market_regime', 'UNKNOWN'),
        datetime.now().isoformat()
    ))
    
    conn.commit()
    conn.close()

def update_ai_decision_outcome(coin, direction, pnl_percent):
    """Update AI decision with actual outcome"""
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE ai_decisions 
        SET outcome_pnl = ?
        WHERE coin = ? AND direction = ? AND outcome_pnl IS NULL
        ORDER BY timestamp DESC
        LIMIT 1
    ''', (pnl_percent, coin, direction))
    
    conn.commit()
    conn.close()

# ========== FEATURE 2: SMART PROFIT TAKING ==========
def check_profit_taking_opportunity(position_data, multi_tf_data):
    """Check if position should take early profits due to trend change"""
    
    current_price = position_data['mark_price']
    entry_price = position_data['entry_price']
    direction = position_data['direction']
    
    if direction == 'LONG':
        pnl_percent = (current_price - entry_price) / entry_price
    else:
        pnl_percent = (entry_price - current_price) / entry_price
    
    # Only consider profit taking if in profit > 1.5%
    if pnl_percent <= 0.015:
        return False, None
    
    # Analyze trend change signals
    trend_signals = []
    signal_strength = 0
    
    # 1. Timeframe alignment check
    tf_alignment = analyze_timeframe_alignment(multi_tf_data)
    
    # If timeframes are no longer aligned with our position
    if direction == 'LONG' and tf_alignment['direction'] in ['bearish', 'strong_bearish']:
        trend_signals.append("Timeframes turned bearish")
        signal_strength += 2
    elif direction == 'SHORT' and tf_alignment['direction'] in ['bullish', 'strong_bullish']:
        trend_signals.append("Timeframes turned bullish")
        signal_strength += 2
    
    # 2. RSI extreme check
    if '1h' in multi_tf_data and 'closes' in multi_tf_data['1h']:
        hourly_closes = multi_tf_data['1h']['closes']
        if len(hourly_closes) > 14:
            current_rsi = calculate_rsi(hourly_closes)
            
            if direction == 'LONG' and current_rsi > 75:
                trend_signals.append(f"RSI overbought ({current_rsi:.0f})")
                signal_strength += 1
            elif direction == 'SHORT' and current_rsi < 25:
                trend_signals.append(f"RSI oversold ({current_rsi:.0f})")
                signal_strength += 1
    
    # Decision logic: Take profits if signal strength >= 2 and profit > 2%
    should_take_profit = signal_strength >= 2 and pnl_percent > 0.02
    
    if should_take_profit:
        reason = f"Smart profit taking: {', '.join(trend_signals)} (Profit: {pnl_percent*100:.1f}%)"
        return True, reason
    
    return False, None

def track_cost(service, cost, units):
    """Track service costs with daily aggregation"""
    global cost_tracker
    
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO cost_tracking (service, cost, units)
        VALUES (?, ?, ?)
    ''', (service, cost, units))
    
    today = datetime.now().date()
    if service == 'openai':
        cursor.execute('''
            INSERT OR REPLACE INTO daily_cost_summary (date, openai_cost, api_calls)
            VALUES (?, 
                    COALESCE((SELECT openai_cost FROM daily_cost_summary WHERE date = ?), 0) + ?,
                    COALESCE((SELECT api_calls FROM daily_cost_summary WHERE date = ?), 0) + 1)
        ''', (today, today, cost, today))
        
        cost_tracker['openai_cost'] += cost
        cost_tracker['api_calls_count'] += 1
    
    conn.commit()
    conn.close()

def get_cost_projections():
    """Calculate cost projections based on historical data"""
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT AVG(openai_cost) as avg_daily_openai,
               AVG(api_calls) as avg_daily_calls
        FROM daily_cost_summary
        WHERE date >= date('now', '-7 days')
    ''')
    
    daily_avg = cursor.fetchone()
    
    hours_running = (datetime.now() - cost_tracker['railway_start_time']).total_seconds() / 3600
    railway_hourly = 0.01
    current_railway_cost = hours_running * railway_hourly
    
    if daily_avg and daily_avg[0]:
        daily_openai = daily_avg[0]
        weekly_openai = daily_openai * 7
        monthly_openai = daily_openai * 30
    else:
        if hours_running > 0:
            daily_openai = (cost_tracker['openai_cost'] / hours_running) * 24
            weekly_openai = daily_openai * 7
            monthly_openai = daily_openai * 30
        else:
            daily_openai = weekly_openai = monthly_openai = 0
    
    weekly_railway = railway_hourly * 24 * 7
    monthly_railway = railway_hourly * 24 * 30
    
    conn.close()
    
    return {
        'current': {
            'openai': cost_tracker['openai_cost'],
            'railway': current_railway_cost,
            'total': cost_tracker['openai_cost'] + current_railway_cost,
            'api_calls': cost_tracker['api_calls_count']
        },
        'projections': {
            'weekly': {
                'openai': weekly_openai,
                'railway': weekly_railway,
                'total': weekly_openai + weekly_railway
            },
            'monthly': {
                'openai': monthly_openai,
                'railway': monthly_railway,
                'total': monthly_openai + monthly_railway
            }
        }
    }

def get_hyperparameter(key, default):
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    cursor.execute('SELECT value FROM hyperparameters WHERE key=?', (key,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else default

def set_hyperparameter(key, value):
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    cursor.execute('INSERT OR REPLACE INTO hyperparameters (key, value) VALUES (?, ?)', (key, value))
    conn.commit()
    conn.close()

# Portfolio management - now uses real Binance data
portfolio = {
    'balance': 0.0,  # Will be fetched from Binance
    'positions': {},  # Will be fetched from Binance
    'trade_history': [],
    'learning_data': {
        'successful_patterns': [],
        'failed_patterns': [],
        'performance_metrics': {},
        'pattern_memory': {}
    }
}

# Flask web app
app = Flask(__name__)

@app.route('/')
def dashboard():
    return send_from_directory('.', 'index.html')

@app.route('/dashboard.js')
def dashboard_js():
    return send_from_directory('.', 'dashboard.js')

@app.route('/api/status')
def api_status():
    try:
        # Get real data from Binance
        portfolio['balance'] = get_futures_balance()
        portfolio['positions'] = get_open_positions()
        
        current_market_data = get_batch_market_data()
        total_value = calculate_portfolio_value(current_market_data)
        learning_insights = get_learning_insights()
        cost_projections = get_cost_projections()
        
        response_data = {
            'total_value': total_value,
            'balance': portfolio['balance'],
            'positions': portfolio['positions'],
            'trade_history': portfolio['trade_history'][-20:],
            'market_data': current_market_data,
            'learning_metrics': learning_insights,
            'cost_tracking': cost_projections,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_flask_app():
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)

def get_batch_market_data():
    """Optimized batch market data fetching"""
    batch_url = "https://api.binance.com/api/v3/ticker/24hr"
    
    try:
        response = requests.get(batch_url)
        all_tickers = response.json()
        
        target_coins = ['BTC', 'ETH', 'SOL', 'BNB', 'SEI', 'DOGE', 
                       'TIA', 'TAO', 'ARB', 'SUI', 'ENA', 'FET']
        
        market_data = {}
        
        for ticker in all_tickers:
            symbol = ticker['symbol']
            if symbol.endswith('USDT'):
                coin = symbol.replace('USDT', '')
                if coin in target_coins:
                    market_data[coin] = {
                        'price': float(ticker['lastPrice']),
                        'change_24h': float(ticker['priceChangePercent']),
                        'volume': float(ticker['volume']),
                        'high_24h': float(ticker['highPrice']),
                        'low_24h': float(ticker['lowPrice']),
                        'quote_volume': float(ticker['quoteVolume'])
                    }
        
        return market_data
        
    except Exception as e:
        print(f"Error fetching batch market data: {e}")
        return {}

def get_multi_timeframe_data(symbol):
    """Get multi-timeframe data for a single symbol"""
    base_url = "https://api.binance.com/api/v3/klines"
    intervals = {
        '5m': ('5m', 48),
        '15m': ('15m', 48),
        '1h': ('1h', 48),
        '4h': ('4h', 42),
        '1d': ('1d', 30)
    }
    
    multi_tf_data = {}
    
    for tf_key, (interval, limit) in intervals.items():
        try:
            response = requests.get(base_url, params={
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            })
            
            if response.status_code == 200:
                candles = response.json()
                multi_tf_data[tf_key] = {
                    'candles': candles,
                    'closes': [float(c[4]) for c in candles],
                    'volumes': [float(c[5]) for c in candles],
                    'highs': [float(c[2]) for c in candles],
                    'lows': [float(c[3]) for c in candles],
                    'opens': [float(c[1]) for c in candles]
                }
        except:
            pass
    
    return multi_tf_data

def calculate_atr(highs, lows, closes, period=14):
    """Calculate Average True Range for volatility-based stops"""
    if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
        return 0
    
    tr_values = []
    for i in range(1, len(highs)):
        high_low = highs[i] - lows[i]
        high_close = abs(highs[i] - closes[i-1])
        low_close = abs(lows[i] - closes[i-1])
        tr = max(high_low, high_close, low_close)
        tr_values.append(tr)
    
    if len(tr_values) < period:
        return 0
    
    atr = sum(tr_values[-period:]) / period
    return atr

def calculate_rsi(prices, period=14):
    """RSI calculation"""
    if len(prices) < period + 1:
        return 50
    
    gains = []
    losses = []
    
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))
    
    if len(gains) < period:
        return 50
        
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_ema(prices, period):
    """Calculate Exponential Moving Average"""
    if len(prices) < period:
        return 0
    
    multiplier = 2 / (period + 1)
    ema = sum(prices[:period]) / period
    
    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema
    
    return ema

def detect_market_regime(multi_tf_data, current_price):
    """Detect if market is trending, ranging, or volatile"""
    if not multi_tf_data or '1h' not in multi_tf_data:
        return {'regime': 'UNKNOWN', 'confidence': 0, 'indicators': {}}
    
    hourly_data = multi_tf_data['1h']
    if 'closes' not in hourly_data or len(hourly_data['closes']) < 20:
        return {'regime': 'UNKNOWN', 'confidence': 0, 'indicators': {}}
    
    closes = hourly_data['closes']
    highs = hourly_data['highs']
    lows = hourly_data['lows']
    
    indicators = {}
    
    # Simple ADX calculation
    def calculate_adx_simplified(highs, lows, closes, period=14):
        if len(highs) < period + 1:
            return 0
        
        plus_dm = []
        minus_dm = []
        tr_values = []
        
        for i in range(1, len(highs)):
            high_diff = highs[i] - highs[i-1]
            low_diff = lows[i-1] - lows[i]
            
            plus_dm.append(max(high_diff, 0) if high_diff > low_diff else 0)
            minus_dm.append(max(low_diff, 0) if low_diff > high_diff else 0)
            
            tr = max(highs[i] - lows[i], 
                    abs(highs[i] - closes[i-1]), 
                    abs(lows[i] - closes[i-1]))
            tr_values.append(tr)
        
        if not tr_values:
            return 0
        
        avg_tr = sum(tr_values[-period:]) / period
        avg_plus_dm = sum(plus_dm[-period:]) / period
        avg_minus_dm = sum(minus_dm[-period:]) / period
        
        if avg_tr == 0:
            return 0
        
        plus_di = (avg_plus_dm / avg_tr) * 100
        minus_di = (avg_minus_dm / avg_tr) * 100
        
        if plus_di + minus_di == 0:
            return 0
        
        dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        return dx
    
    adx = calculate_adx_simplified(highs, lows, closes)
    indicators['adx'] = adx
    
    # Moving averages
    sma_20 = sum(closes[-20:]) / 20
    sma_50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else sma_20
    
    price_above_sma = current_price > sma_20
    price_above_sma50 = current_price > sma_50
    
    # Determine regime
    confidence = 0
    regime = 'UNKNOWN'
    
    if adx > 25:  # Strong trend
        if price_above_sma and price_above_sma50:
            regime = 'TRENDING_UP'
            confidence = min(adx / 40 * 100, 90)
        elif not price_above_sma and not price_above_sma50:
            regime = 'TRENDING_DOWN'
            confidence = min(adx / 40 * 100, 90)
    elif adx < 20:  # Ranging market
        regime = 'RANGING'
        confidence = 70
    else:
        regime = 'TRANSITIONAL'
        confidence = 50
    
    return {
        'regime': regime,
        'confidence': confidence,
        'indicators': indicators
    }

def analyze_timeframe_alignment(multi_tf_data):
    """Enhanced timeframe alignment analysis"""
    if not multi_tf_data:
        return {'aligned': False, 'score': 0, 'direction': 'neutral', 'signals': {}, 'strength': 0}
    
    alignment_score = 0
    signals = {}
    strength_scores = []
    
    for tf, data in multi_tf_data.items():
        if 'closes' in data and len(data['closes']) > 20:
            closes = data['closes']
            
            sma_short = sum(closes[-5:]) / 5
            sma_mid = sum(closes[-10:]) / 10
            sma_long = sum(closes[-20:]) / 20
            
            tf_score = 0
            
            if sma_short > sma_mid > sma_long:
                tf_score += 2
                signals[tf] = 'strong_bullish'
            elif sma_short > sma_mid and sma_mid > sma_long * 0.995:
                tf_score += 1
                signals[tf] = 'bullish'
            elif sma_short < sma_mid < sma_long:
                tf_score -= 2
                signals[tf] = 'strong_bearish'
            elif sma_short < sma_mid and sma_mid < sma_long * 1.005:
                tf_score -= 1
                signals[tf] = 'bearish'
            else:
                signals[tf] = 'neutral'
            
            alignment_score += tf_score
            strength_scores.append(abs(tf_score))
    
    avg_strength = sum(strength_scores) / len(strength_scores) if strength_scores else 0
    
    if alignment_score >= 4:
        direction = 'strong_bullish'
        aligned = True
    elif alignment_score >= 2:
        direction = 'bullish'
        aligned = True
    elif alignment_score <= -4:
        direction = 'strong_bearish'
        aligned = True
    elif alignment_score <= -2:
        direction = 'bearish'
        aligned = True
    else:
        direction = 'mixed'
        aligned = False
    
    return {
        'aligned': aligned,
        'direction': direction,
        'score': alignment_score,
        'signals': signals,
        'strength': avg_strength
    }

def calculate_dynamic_stops(current_price, atr, direction, market_regime):
    """Calculate dynamic stop loss and take profit based on market conditions"""
    
    sl_multiplier = get_hyperparameter('sl_multiplier', 1.5)
    tp_multiplier = get_hyperparameter('tp_multiplier', 2.5)
    
    if market_regime == 'VOLATILE':
        sl_multiplier = 2.0
        tp_multiplier = 3.5
    elif market_regime == 'RANGING':
        sl_multiplier = 1.0
        tp_multiplier = 1.5
    elif market_regime in ['TRENDING_UP', 'TRENDING_DOWN']:
        sl_multiplier = 1.8
        tp_multiplier = 4.0
    
    atr_stop_distance = atr * sl_multiplier
    atr_target_distance = atr * tp_multiplier
    
    if direction == 'LONG':
        stop_loss = current_price - atr_stop_distance
        take_profit = current_price + atr_target_distance
    else:  # SHORT
        stop_loss = current_price + atr_stop_distance
        take_profit = current_price - atr_target_distance
    
    sl_percentage = abs(stop_loss - current_price) / current_price
    tp_percentage = abs(take_profit - current_price) / current_price
    
    # Ensure minimum and maximum boundaries
    min_sl = 0.01
    max_sl = 0.05
    min_tp = 0.015
    max_tp = 0.10
    
    if sl_percentage < min_sl:
        if direction == 'LONG':
            stop_loss = current_price * (1 - min_sl)
        else:
            stop_loss = current_price * (1 + min_sl)
    elif sl_percentage > max_sl:
        if direction == 'LONG':
            stop_loss = current_price * (1 - max_sl)
        else:
            stop_loss = current_price * (1 + max_sl)
    
    if tp_percentage < min_tp:
        if direction == 'LONG':
            take_profit = current_price * (1 + min_tp)
        else:
            take_profit = current_price * (1 - min_tp)
    elif tp_percentage > max_tp:
        if direction == 'LONG':
            take_profit = current_price * (1 + max_tp)
        else:
            take_profit = current_price * (1 - max_tp)
    
    return {
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'sl_percentage': abs(stop_loss - current_price) / current_price * 100,
        'tp_percentage': abs(take_profit - current_price) / current_price * 100,
        'risk_reward': tp_percentage / sl_percentage if sl_percentage > 0 else 0
    }

def ai_trade_decision(coin, market_data, multi_tf_data, learning_insights):
    """Enhanced AI trading decision with memory context"""
    if not client:
        return None
    
    current_price = market_data[coin]['price']
    tf_alignment = analyze_timeframe_alignment(multi_tf_data)
    
    if not tf_alignment['aligned']:
        return {'direction': 'SKIP', 'reason': 'Timeframes not aligned'}
    
    # Calculate advanced indicators
    hourly_data = multi_tf_data.get('1h', {})
    
    atr = 0
    if 'highs' in hourly_data and 'lows' in hourly_data and 'closes' in hourly_data:
        atr = calculate_atr(hourly_data['highs'], hourly_data['lows'], hourly_data['closes'])
    
    market_regime = detect_market_regime(multi_tf_data, current_price)
    
    dynamic_stops = calculate_dynamic_stops(
        current_price, 
        atr, 
        'LONG' if tf_alignment['direction'] in ['bullish', 'strong_bullish'] else 'SHORT',
        market_regime['regime']
    )
    
    # Get AI memory context
    ai_context = get_ai_context_memory()
    
    # Enhanced prompt
    analysis_prompt = f"""
{ai_context}

---

{coin} ADVANCED TRADING ANALYSIS

Price: ${current_price:.2f}
24h: {market_data[coin]['change_24h']:+.1f}%
Volume: {market_data[coin]['volume']/1000000:.1f}M

MARKET STRUCTURE:
- Regime: {market_regime['regime']} (confidence: {market_regime['confidence']:.0f}%)
- Timeframe Alignment: {tf_alignment['direction']} (score: {tf_alignment['score']:.1f})
- ATR: ${atr:.2f}

DYNAMIC RISK:
- Suggested SL: {dynamic_stops['sl_percentage']:.1f}%
- Suggested TP: {dynamic_stops['tp_percentage']:.1f}%
- Risk/Reward: {dynamic_stops['risk_reward']:.1f}

LEARNING:
- Win Rate: {learning_insights.get('win_rate', 0):.0f}%
- Best Leverage: {learning_insights.get('best_leverage', 10)}x

Based on your recent performance and market structure, provide:
DECISION: [LONG/SHORT/SKIP]
LEVERAGE: [10-30] (higher for strong confidence)
SIZE: [5-15]%
CONFIDENCE: [1-10]
REASON: [one line about key factors]
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": analysis_prompt}],
            max_tokens=150,
            temperature=0.4
        )
        
        track_cost('openai', 0.00008, '300')
        
        ai_response = response.choices[0].message.content.strip()
        trade_params = {}
        
        for line in ai_response.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().upper()
                value = value.strip()
                
                if 'DECISION' in key:
                    trade_params['direction'] = value.upper()
                elif 'LEVERAGE' in key:
                    trade_params['leverage'] = min(30, max(10, int(''.join(filter(str.isdigit, value)))))
                elif 'SIZE' in key:
                    size = float(''.join(filter(lambda x: x.isdigit() or x == '.', value)))
                    trade_params['position_size'] = min(0.15, max(0.05, size / 100))
                elif 'CONFIDENCE' in key:
                    trade_params['confidence'] = min(10, max(1, int(''.join(filter(str.isdigit, value)))))
                elif 'REASON' in key:
                    trade_params['reasoning'] = value[:150]
        
        trade_params['stop_loss'] = dynamic_stops['stop_loss']
        trade_params['take_profit'] = dynamic_stops['take_profit']
        trade_params['sl_percentage'] = dynamic_stops['sl_percentage']
        trade_params['tp_percentage'] = dynamic_stops['tp_percentage']
        trade_params['market_regime'] = market_regime['regime']
        trade_params['atr'] = atr
        
        # Default values
        if 'direction' not in trade_params:
            trade_params['direction'] = 'SKIP'
        if 'leverage' not in trade_params:
            trade_params['leverage'] = 15
        if 'position_size' not in trade_params:
            trade_params['position_size'] = 0.08
        if 'confidence' not in trade_params:
            trade_params['confidence'] = 5
        if 'reasoning' not in trade_params:
            trade_params['reasoning'] = f"{market_regime['regime']} market, {tf_alignment['direction']} alignment"
        
        return trade_params
        
    except Exception as e:
        print(f"AI Error: {e}")
        return None

def get_symbol_precision(symbol):
    """Get precision info for symbol from Binance"""
    try:
        exchange_info = requests.get(f"{BINANCE_BASE_URL}/fapi/v1/exchangeInfo")
        if exchange_info.status_code == 200:
            data = exchange_info.json()
            for symbol_info in data['symbols']:
                if symbol_info['symbol'] == symbol:
                    # Get quantity precision
                    for filter_info in symbol_info['filters']:
                        if filter_info['filterType'] == 'LOT_SIZE':
                            step_size = float(filter_info['stepSize'])
                            # Calculate decimal places from step size
                            if step_size >= 1:
                                return 0
                            else:
                                return len(str(step_size).split('.')[-1].rstrip('0'))
        return 3  # Default fallback
    except:
        return 3  # Default fallback

def execute_real_trade(coin, trade_params, current_price):
    """Execute real trade on Binance futures"""
    if trade_params['direction'] == 'SKIP':
        return
    
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        symbol = f"{coin}USDT"
        
        # Calculate position size in USDT
        balance = get_futures_balance()
        if balance < 5:  # Minimum balance check
            print(f"❌ Insufficient balance: ${balance:.2f}")
            return
        
        position_value = balance * trade_params.get('position_size', 0.08)
        leverage = trade_params['leverage']
        
        # Calculate quantity
        notional_value = position_value * leverage
        quantity = notional_value / current_price
        
        # Get proper precision for this symbol
        precision = get_symbol_precision(symbol)
        quantity = round(quantity, precision)
        
        # Additional precision fixes for major coins
        if coin in ['BTC']:
            quantity = round(quantity, 3)  # BTC: 3 decimals max
        elif coin in ['ETH']:
            quantity = round(quantity, 2)  # ETH: 2 decimals max
        elif coin in ['SOL', 'BNB']:
            quantity = round(quantity, 1)  # SOL/BNB: 1 decimal max
        else:
            quantity = round(quantity, 0)  # Others: whole numbers
        
        if quantity * current_price < 5:  # Binance minimum notional
            print(f"❌ Position too small: ${quantity * current_price:.2f}")
            return
        
        print(f"   📊 Trade Details:")
        print(f"      Quantity: {quantity} {coin}")
        print(f"      Notional: ${quantity * current_price:.2f}")
        print(f"      Precision: {precision} decimals")
        
        # Place order
        side = "BUY" if trade_params['direction'] == 'LONG' else "SELL"
        
        order_result = place_futures_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            leverage=leverage
        )
        
        if order_result:
            print(f"\n✅ REAL TRADE EXECUTED:")
            print(f"   {trade_params['direction']} {coin}: {quantity} @ {leverage}x")
            print(f"   Value: ${position_value:.2f} | Notional: ${notional_value:.2f}")
            print(f"   Confidence: {trade_params['confidence']}/10")
            print(f"   Reason: {trade_params['reasoning'][:50]}...")
            
            # Set stop loss and take profit
            set_stop_loss_take_profit(
                symbol,
                trade_params['stop_loss'],
                trade_params['take_profit'],
                trade_params['direction']
            )
            
            # Save AI decision for memory
            save_ai_decision(coin, trade_params)
            
            # Save to database
            save_trade_to_db({
                'time': timestamp,
                'position_id': f"{symbol}_{timestamp.replace(' ', '_').replace(':', '-')}",
                'coin': coin,
                'action': f"{trade_params['direction']} OPEN",
                'direction': trade_params['direction'],
                'price': current_price,
                'position_size': position_value,
                'leverage': leverage,
                'notional_value': notional_value,
                'stop_loss': trade_params['stop_loss'],
                'take_profit': trade_params['take_profit'],
                'confidence': trade_params['confidence'],
                'reason': trade_params['reasoning']
            })
        else:
            print(f"❌ Failed to execute trade for {coin}")
            
    except Exception as e:
        print(f"❌ Trade execution error: {e}")

def monitor_positions():
    """Monitor real positions from Binance"""
    try:
        positions = get_open_positions()
        
        for symbol, position_data in positions.items():
            coin = position_data['coin']
            
            # Get multi-timeframe data for analysis
            multi_tf_data = get_multi_timeframe_data(symbol)
            
            # Check for smart profit taking
            should_close, reason = check_profit_taking_opportunity(position_data, multi_tf_data)
            
            if should_close:
                print(f"🧠 Smart exit triggered for {coin}: {reason}")
                close_result = close_futures_position(symbol)
                
                if close_result:
                    print(f"✅ Position closed: {coin}")
                    
                    # Update AI decision outcome
                    pnl_percent = (position_data['pnl'] / (position_data['notional'] / position_data['leverage'])) * 100
                    update_ai_decision_outcome(coin, position_data['direction'], pnl_percent)
                else:
                    print(f"❌ Failed to close position: {coin}")
        
        return positions
        
    except Exception as e:
        print(f"Error monitoring positions: {e}")
        return {}

def save_trade_to_db(trade_record):
    """Save trade to database"""
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT OR REPLACE INTO trades 
        (timestamp, position_id, coin, action, direction, price, position_size, 
         leverage, notional_value, stop_loss, take_profit, pnl, pnl_percent, 
         duration, reason, confidence, profitable)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        trade_record.get('time'),
        trade_record.get('position_id'),
        trade_record.get('coin'),
        trade_record.get('action'),
        trade_record.get('direction'),
        trade_record.get('price'),
        trade_record.get('position_size'),
        trade_record.get('leverage'),
        trade_record.get('notional_value'),
        trade_record.get('stop_loss'),
        trade_record.get('take_profit'),
        trade_record.get('pnl'),
        trade_record.get('pnl_percent'),
        trade_record.get('duration'),
        trade_record.get('reason'),
        trade_record.get('confidence'),
        trade_record.get('pnl', 0) > 0 if 'pnl' in trade_record else None
    ))
    
    conn.commit()
    conn.close()

def get_learning_insights():
    """Get learning insights from database"""
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    
    insights = {}
    
    cursor.execute('SELECT COUNT(*) FROM trades WHERE pnl IS NOT NULL')
    total = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM trades WHERE pnl > 0')
    wins = cursor.fetchone()[0]
    
    insights['win_rate'] = (wins / total * 100) if total > 0 else 0
    insights['total_trades'] = total
    
    cursor.execute('''
        SELECT leverage, AVG(pnl_percent) as avg_return
        FROM trades 
        WHERE pnl IS NOT NULL 
        GROUP BY leverage 
        ORDER BY avg_return DESC 
        LIMIT 1
    ''')
    
    best_lev = cursor.fetchone()
    insights['best_leverage'] = best_lev[0] if best_lev else 15
    
    cursor.execute('SELECT AVG(pnl) FROM trades WHERE pnl > 0')
    avg_profit = cursor.fetchone()[0]
    insights['avg_profit'] = avg_profit if avg_profit else 0
    
    cursor.execute('SELECT AVG(pnl) FROM trades WHERE pnl < 0')
    avg_loss = cursor.fetchone()[0]
    insights['avg_loss'] = avg_loss if avg_loss else 0
    
    conn.close()
    return insights

def calculate_portfolio_value(market_data):
    """Calculate total portfolio value from real Binance data"""
    balance = get_futures_balance()
    positions = get_open_positions()
    
    total_pnl = sum(pos['pnl'] for pos in positions.values())
    
    return balance + total_pnl

def run_enhanced_bot():
    """Main bot loop with real Binance trading"""
    init_database()
    
    print("🚀 REAL BINANCE FUTURES TRADING BOT - LIVE MODE")
    print("🔥 Features: AI Memory + Smart Profit Taking + Real Trading")
    print("⚡ Leverage: 10x-30x based on confidence")
    print("💰 Balance Detection: Auto-detects USDC/BNFCR balance")
    print("📊 Works with ANY balance (minimum $5)")
    print("🎯 Coins: BTC, ETH, SOL, BNB, SEI, DOGE, TIA, TAO, ARB, SUI, ENA, FET")
    print("="*80)
    
    # Test Binance connection
    balance = get_futures_balance()
    if balance <= 0:
        print("❌ No futures balance detected or API connection failed")
        return
    
    print(f"✅ Connected to Binance Futures | Balance: ${balance:.2f}")
    
    target_coins = ['BTC', 'ETH', 'SOL', 'BNB', 'SEI', 'DOGE', 
                   'TIA', 'TAO', 'ARB', 'SUI', 'ENA', 'FET']
    
    price_histories = {coin: [] for coin in target_coins}
    last_full_analysis = 0
    iteration = 0
    
    while True:
        try:
            iteration += 1
            current_time = time.time()
            
            # Quick position monitoring
            current_positions = monitor_positions()
            
            # Full analysis every 3 minutes (increased from 2 minutes)
            if current_time - last_full_analysis >= 180:  # 3 minutes = 180 seconds
                print(f"\n{'='*80}")
                print(f"🧠 LIVE MARKET ANALYSIS #{iteration}")
                print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("="*80)
                
                # Get fresh balance and positions
                current_balance = get_futures_balance()
                portfolio['balance'] = current_balance
                portfolio['positions'] = current_positions
                
                market_data = get_batch_market_data()
                
                # Update price histories
                for coin in target_coins:
                    if coin in market_data:
                        price_histories[coin].append(market_data[coin]['price'])
                        if len(price_histories[coin]) > 100:
                            price_histories[coin] = price_histories[coin][-100:]
                
                # Market overview
                print(f"\n💰 Account Status:")
                print(f"   Balance: ${current_balance:.2f}")
                print(f"   Open Positions: {len(current_positions)}")
                
                if current_positions:
                    total_pnl = sum(pos['pnl'] for pos in current_positions.values())
                    print(f"   Unrealized P&L: ${total_pnl:+.2f}")
                
                print(f"\n📊 Market Overview:")
                for coin in target_coins[:6]:
                    if coin in market_data:
                        data = market_data[coin]
                        symbol = f"{coin}USDT"
                        coin_tf_data = get_multi_timeframe_data(symbol)
                        
                        rsi = calculate_rsi(price_histories[coin]) if len(price_histories[coin]) > 14 else 50
                        regime_data = detect_market_regime(coin_tf_data, data['price'])
                        regime = regime_data['regime']
                        
                        if regime == 'TRENDING_UP':
                            trend = "🚀"
                        elif regime == 'TRENDING_DOWN':
                            trend = "📉"
                        elif regime == 'RANGING':
                            trend = "↔️"
                        else:
                            trend = "❓"
                        
                        position_info = ""
                        symbol_check = f"{coin}USDT"
                        if symbol_check in current_positions:
                            pos = current_positions[symbol_check]
                            position_info = f" [{pos['direction']} {pos['pnl']:+.1f}]"
                        
                        print(f"   {coin}: ${data['price']:.2f} {trend} ({data['change_24h']:+.1f}%) RSI:{rsi:.0f}{position_info}")
                
                # Look for new opportunities
                if len(current_positions) < MAX_CONCURRENT_POSITIONS and current_balance >= 5:
                    learning_insights = get_learning_insights()
                    
                    print(f"\n🤖 AI Comprehensive Analysis & Opportunity Scan:")
                    print(f"   Historical Win Rate: {learning_insights['win_rate']:.1f}%")
                    print(f"   Optimal Leverage: {learning_insights.get('best_leverage', 15)}x")
                    print(f"   Analyzing market conditions...")
                    
                    # Get AI context for decision making
                    ai_context = get_ai_context_memory()
                    if "win rate" in ai_context.lower():
                        recent_performance = ai_context.split("RECENT PERFORMANCE:")[1].split("\n")[0].strip() if "RECENT PERFORMANCE:" in ai_context else ""
                        if recent_performance:
                            print(f"   Recent AI Performance: {recent_performance}")
                    
                    opportunities_found = 0
                    opportunities_analyzed = 0
                    
                    for coin in target_coins:
                        # Skip if already have position
                        symbol_check = f"{coin}USDT"
                        if symbol_check in current_positions:
                            continue
                        
                        if opportunities_found >= 1:  # Limit to 1 trade per cycle for safety
                            print(f"   🛡️ Safety limit: 1 trade per cycle")
                            break
                        
                        opportunities_analyzed += 1
                        print(f"   🔍 Analyzing {coin}...")
                        
                        symbol = f"{coin}USDT"
                        multi_tf_data = get_multi_timeframe_data(symbol)
                        
                        # Market regime analysis
                        regime_data = detect_market_regime(multi_tf_data, market_data[coin]['price'])
                        print(f"      Market Regime: {regime_data['regime']} (confidence: {regime_data['confidence']:.0f}%)")
                        
                        # Timeframe alignment check
                        tf_alignment = analyze_timeframe_alignment(multi_tf_data)
                        print(f"      Timeframe Alignment: {tf_alignment['direction']} (score: {tf_alignment['score']:.1f})")
                        
                        if regime_data['regime'] == 'TRANSITIONAL' and regime_data['confidence'] < 60:
                            print(f"      ⏭️ Skipping: Market too transitional")
                            continue
                        
                        if not tf_alignment['aligned']:
                            print(f"      ⏭️ Skipping: Timeframes not aligned")
                            continue
                        
                        # AI decision with full analysis
                        print(f"      🧠 Running AI analysis with memory context...")
                        trade_params = ai_trade_decision(coin, market_data, multi_tf_data, learning_insights)
                        
                        if trade_params and trade_params['direction'] != 'SKIP':
                            print(f"      AI Decision: {trade_params['direction']} (Confidence: {trade_params['confidence']}/10)")
                            print(f"      Reasoning: {trade_params['reasoning'][:60]}...")
                            
                            if trade_params['confidence'] >= MIN_CONFIDENCE_THRESHOLD:
                                risk_reward = trade_params.get('tp_percentage', 0) / trade_params.get('sl_percentage', 1)
                                print(f"      Risk/Reward: 1:{risk_reward:.1f}")
                                
                                if risk_reward >= 2.0:  # Increased minimum risk/reward to 2:1
                                    print(f"   ✅ HIGH-CONFIDENCE OPPORTUNITY FOUND!")
                                    print(f"      {trade_params['direction']} {coin} | Confidence: {trade_params['confidence']}/10 | Leverage: {trade_params['leverage']}x")
                                    
                                    execute_real_trade(coin, trade_params, market_data[coin]['price'])
                                    opportunities_found += 1
                                    
                                    # Wait between trades
                                    time.sleep(2)
                                else:
                                    print(f"      ⏭️ Poor risk/reward ratio (1:{risk_reward:.1f})")
                            else:
                                print(f"      ⏭️ Low confidence ({trade_params['confidence']}/10 < {MIN_CONFIDENCE_THRESHOLD})")
                        else:
                            print(f"      ⏭️ AI recommends SKIP")
                    
                    if opportunities_found == 0:
                        print(f"   📊 Analysis Complete: No high-confidence opportunities from {opportunities_analyzed} coins")
                        print(f"      Waiting for better market conditions...")
                    else:
                        print(f"   🎯 Executed {opportunities_found} trades from {opportunities_analyzed} analyzed coins")
                else:
                    if current_balance < 5:
                        print(f"\n⚠️ Balance too low: ${current_balance:.2f}")
                    else:
                        print(f"\n⚠️ Position limit reached ({len(current_positions)}/{MAX_CONCURRENT_POSITIONS})")
                
                # Performance summary
                total_value = calculate_portfolio_value(market_data)
                cost_data = get_cost_projections()
                
                print(f"\n📈 Performance:")
                print(f"   Total Value: ${total_value:.2f}")
                print(f"   Session Costs: ${cost_data['current']['total']:.4f}")
                
                last_full_analysis = current_time
                print("="*80)
            
            time.sleep(QUICK_CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            print("\n\n🛑 Bot stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(QUICK_CHECK_INTERVAL)

if __name__ == "__main__":
    # Start Flask
    flask_thread = threading.Thread(target=run_flask_app, daemon=True)
    flask_thread.start()
    
    # Start real trading bot
    run_enhanced_bot()