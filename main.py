import requests
import time
import os
from openai import OpenAI
import statistics
import json
from datetime import datetime, timedelta
from flask import Flask, jsonify, send_from_directory
import threading
import math
import sqlite3
from collections import deque
import hmac
import hashlib
import urllib.parse

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None


# Cost tracking with projections
cost_tracker = {
    'openai_tokens': 0,
    'openai_cost': 0.0,  # GPT-4o-mini: $0.15/1M input, $0.60/1M output
    'api_calls_count': 0,
    'railway_start_time': datetime.now(),
    'daily_costs': [],  # Track daily costs for projections
    'last_reset': datetime.now()
}

# Constants for optimization
QUICK_CHECK_INTERVAL = 15  # Check positions every 15 seconds
FULL_ANALYSIS_INTERVAL = 120  # Full market analysis every 2 minutes
MAX_CONCURRENT_POSITIONS = 4  # Limit concurrent positions
MIN_CONFIDENCE_THRESHOLD = 6  # Only take trades with confidence >= 6
BATCH_API_WEIGHT_LIMIT = 100  # Binance API weight limit per batch
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY") 
FUTURES_BASE_URL = "https://fapi.binance.com"
USE_REAL_TRADING = True

if USE_REAL_TRADING:
    if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
        print("❌ ERROR: Real trading enabled but API keys not found!")
        print("Set BINANCE_API_KEY and BINANCE_SECRET_KEY environment variables")
        exit(1)
    else:
        print(f"✅ Real trading enabled with API key: {BINANCE_API_KEY[:8]}...")

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
    
    # Learning patterns table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS learning_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            coin TEXT NOT NULL,
            direction TEXT NOT NULL,
            leverage INTEGER NOT NULL,
            confidence INTEGER NOT NULL,
            duration_target TEXT NOT NULL,
            profitable BOOLEAN NOT NULL,
            pnl_percent REAL NOT NULL,
            duration_actual TEXT,
            market_conditions TEXT,
            rsi_entry REAL,
            rsi_exit REAL,
            volume_ratio REAL,
            sentiment_score REAL,
            timeframe_alignment TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Pattern recognition table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pattern_recognition (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_type TEXT NOT NULL,
            coin TEXT NOT NULL,
            success_rate REAL NOT NULL,
            occurrences INTEGER NOT NULL,
            avg_return REAL,
            conditions TEXT,
            last_seen DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Market regime tracking
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_regimes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            coin TEXT NOT NULL,
            regime TEXT NOT NULL,
            confidence REAL NOT NULL,
            indicators TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Cost tracking table with daily aggregation
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

    # Hyperparameters table for auto-tuning
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS hyperparameters (
            key TEXT PRIMARY KEY,
            value REAL NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()
    print("📁 Enhanced SQLite database initialized")

def generate_signature(params, secret):
    """Generate signature for Binance API"""
    query_string = urllib.parse.urlencode(params)
    return hmac.new(secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()

def get_real_balance():
    """Get real USD balance from Binance Futures (USDT)"""
    if not USE_REAL_TRADING or not BINANCE_API_KEY:
        return portfolio['balance']
    
    try:
        # Use correct futures balance endpoint
        endpoint = "/fapi/v2/balance"
        timestamp = int(time.time() * 1000)
        
        params = {'timestamp': timestamp}
        signature = generate_signature(params, BINANCE_SECRET_KEY)
        params['signature'] = signature
        
        headers = {'X-MBX-APIKEY': BINANCE_API_KEY}
        response = requests.get(FUTURES_BASE_URL + endpoint, params=params, headers=headers, timeout=10)
        
        print(f"Balance API Response Status: {response.status_code}")
        
        if response.status_code == 200:
            balances = response.json()
            print(f"Raw balance response: {balances}")
            
            for balance in balances:
                # Look for USDT balance specifically
                if balance['asset'] in ['USDC', 'BNFCR']:
                    available_balance = float(balance['balance'])
                    wallet_balance = float(balance.get('walletBalance', balance['balance']))
                    
                    print(f"USDT Available: ${available_balance:.2f}, Wallet: ${wallet_balance:.2f}")
                    
                    # Use wallet balance (total) rather than available balance
                    if wallet_balance > 0:
                        return wallet_balance
                    elif available_balance > 0:
                        return available_balance
            
            print("No USDT balance found in futures account")
            return portfolio['balance']
        else:
            print(f"Balance API Error: {response.status_code} - {response.text}")
            return portfolio['balance']
            
    except Exception as e:
        print(f"Error getting real balance: {e}")
        return portfolio['balance']
    
def get_account_info():
    """Get complete Binance Futures account information"""
    if not USE_REAL_TRADING or not BINANCE_API_KEY:
        return None
    
    try:
        endpoint = "/fapi/v2/account"
        timestamp = int(time.time() * 1000)
        
        params = {'timestamp': timestamp}
        signature = generate_signature(params, BINANCE_SECRET_KEY)
        params['signature'] = signature
        
        headers = {'X-MBX-APIKEY': BINANCE_API_KEY}
        response = requests.get(FUTURES_BASE_URL + endpoint, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            account_info = response.json()
            print(f"Account Info - Total Wallet Balance: ${account_info.get('totalWalletBalance', 0)}")
            print(f"Account Info - Available Balance: ${account_info.get('availableBalance', 0)}")
            print(f"Account Info - Total Unrealized PnL: ${account_info.get('totalUnrealizedPnL', 0)}")
            return account_info
        else:
            print(f"Account Info Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Error getting account info: {e}")
        return None
    
def sync_real_balance():
    """Synchronize portfolio balance with real Binance balance"""
    if not USE_REAL_TRADING:
        print("📝 Paper trading mode - using simulated balance")
        return
    
    print("💰 Syncing with Binance Futures account...")
    
    # Get detailed account info first
    account_info = get_account_info()
    if account_info:
        total_wallet = float(account_info.get('totalWalletBalance', 0))
        available_balance = float(account_info.get('availableBalance', 0))
        unrealized_pnl = float(account_info.get('totalUnrealizedPnL', 0))
        
        print(f"   📊 Wallet Balance: ${total_wallet:.2f}")
        print(f"   💵 Available: ${available_balance:.2f}")
        print(f"   📈 Unrealized PnL: ${unrealized_pnl:+.2f}")
        
        # Update portfolio with real balance
        portfolio['balance'] = available_balance
        print(f"   ✅ Portfolio balance updated to: ${available_balance:.2f}")
        
        # Check for existing positions
        existing_positions = account_info.get('positions', [])
        active_positions = [pos for pos in existing_positions if float(pos.get('positionAmt', 0)) != 0]
        
        if active_positions:
            print(f"   ⚠️  Found {len(active_positions)} existing positions on Binance:")
            for pos in active_positions:
                symbol = pos.get('symbol', '')
                amount = float(pos.get('positionAmt', 0))
                entry_price = float(pos.get('entryPrice', 0))
                pnl = float(pos.get('unRealizedPnl', 0))
                
                side = "LONG" if amount > 0 else "SHORT"
                print(f"      • {symbol}: {side} {abs(amount):.4f} @ ${entry_price:.2f} (PnL: ${pnl:+.2f})")
    else:
        # Fallback to simple balance check
        real_balance = get_real_balance()
        if real_balance != portfolio['balance']:
            portfolio['balance'] = real_balance
            print(f"   ✅ Balance updated to: ${real_balance:.2f}")

def place_real_order(symbol, side, quantity, leverage):
    """Place real futures order"""
    if not USE_REAL_TRADING or not BINANCE_API_KEY:
        return {'orderId': 'PAPER_TRADE', 'status': 'FILLED'}
    
    try:
        # Set leverage first
        set_leverage(symbol, leverage)
        
        # Place market order
        endpoint = "/fapi/v1/order"
        timestamp = int(time.time() * 1000)
        
        params = {
            'symbol': symbol,
            'side': side,
            'type': 'MARKET',
            'quantity': f"{quantity:.3f}",
            'timestamp': timestamp
        }
        
        signature = generate_signature(params, BINANCE_SECRET_KEY)
        params['signature'] = signature
        
        headers = {'X-MBX-APIKEY': BINANCE_API_KEY}
        response = requests.post(FUTURES_BASE_URL + endpoint, params=params, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Order failed: {response.text}")
            return None
    except Exception as e:
        print(f"Error placing order: {e}")
        return None

def set_leverage(symbol, leverage):
    """Set leverage for symbol"""
    if not USE_REAL_TRADING or not BINANCE_API_KEY:
        return True
    
    try:
        endpoint = "/fapi/v1/leverage"
        timestamp = int(time.time() * 1000)
        
        params = {
            'symbol': symbol,
            'leverage': leverage,
            'timestamp': timestamp
        }
        
        signature = generate_signature(params, BINANCE_SECRET_KEY)
        params['signature'] = signature
        
        headers = {'X-MBX-APIKEY': BINANCE_API_KEY}
        response = requests.post(FUTURES_BASE_URL + endpoint, params=params, headers=headers)
        return response.status_code == 200
    except Exception as e:
        print(f"Error setting leverage: {e}")
        return False

def close_real_position(symbol, side, quantity):
    """Close real position"""
    if not USE_REAL_TRADING or not BINANCE_API_KEY:
        return {'orderId': 'PAPER_CLOSE', 'status': 'FILLED'}
    
    try:
        endpoint = "/fapi/v1/order"
        timestamp = int(time.time() * 1000)
        
        params = {
            'symbol': symbol,
            'side': side,
            'type': 'MARKET',
            'quantity': f"{quantity:.3f}",
            'timestamp': timestamp
        }
        
        signature = generate_signature(params, BINANCE_SECRET_KEY)
        params['signature'] = signature
        
        headers = {'X-MBX-APIKEY': BINANCE_API_KEY}
        response = requests.post(FUTURES_BASE_URL + endpoint, params=params, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"Error closing position: {e}")
        return None

def track_cost(service, cost, units):
    """Track service costs with daily aggregation"""
    global cost_tracker
    
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    
    # Save individual cost
    cursor.execute('''
        INSERT INTO cost_tracking (service, cost, units)
        VALUES (?, ?, ?)
    ''', (service, cost, units))
    
    # Update daily summary
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
    
    # Get last 7 days of costs
    cursor.execute('''
        SELECT AVG(openai_cost) as avg_daily_openai,
               AVG(api_calls) as avg_daily_calls
        FROM daily_cost_summary
        WHERE date >= date('now', '-7 days')
    ''')
    
    daily_avg = cursor.fetchone()
    
    # Calculate runtime cost
    hours_running = (datetime.now() - cost_tracker['railway_start_time']).total_seconds() / 3600
    railway_hourly = 0.01  # $0.01/hour estimate
    current_railway_cost = hours_running * railway_hourly
    
    # Projections
    if daily_avg and daily_avg[0]:
        daily_openai = daily_avg[0]
        weekly_openai = daily_openai * 7
        monthly_openai = daily_openai * 30
    else:
        # Estimate based on current session
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

def save_active_position(position_id, position):
    """Save active position to database"""
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT OR REPLACE INTO active_positions 
        (position_id, coin, direction, entry_price, entry_time, position_size,
         leverage, notional_value, stop_loss, take_profit, duration_target,
         confidence, reasoning, market_regime, atr_at_entry)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        position_id,
        position['coin'],
        position['direction'],
        position['entry_price'],
        position['entry_time'],
        position['position_size'],
        position['leverage'],
        position['notional_value'],
        position['stop_loss'],
        position['take_profit'],
        position.get('duration_target', 'SWING'),
        position.get('confidence', 5),
        position.get('reasoning', ''),
        position.get('market_regime', 'UNKNOWN'),
        position.get('atr', 0)
    ))
    
    conn.commit()
    conn.close()

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

def load_active_positions():
    """Load active positions from database"""
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM active_positions')
    positions = cursor.fetchall()
    
    active_positions = {}
    for pos in positions:
        position_id = pos[0]
        active_positions[position_id] = {
            'coin': pos[1],
            'direction': pos[2],
            'entry_price': pos[3],
            'entry_time': pos[4],
            'position_size': pos[5],
            'leverage': pos[6],
            'notional_value': pos[7],
            'stop_loss': pos[8],
            'take_profit': pos[9],
            'duration_target': pos[10],
            'confidence': pos[11],
            'reasoning': pos[12],
            'market_regime': pos[13] if len(pos) > 13 else 'UNKNOWN',
            'atr': pos[14] if len(pos) > 14 else 0
        }
    
    conn.close()
    return active_positions

def remove_active_position(position_id):
    """Remove closed position from active positions table"""
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM active_positions WHERE position_id = ?', (position_id,))
    conn.commit()
    conn.close()

# Portfolio management
portfolio = {
    'balance': 1000.0,
    'positions': {},
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
        # Sync balance before sending status
        if USE_REAL_TRADING:
            real_balance = get_real_balance()
            if real_balance != portfolio['balance']:
                portfolio['balance'] = real_balance
        
        current_market_data = get_batch_market_data()
        total_value = calculate_portfolio_value(current_market_data)
        learning_insights = get_learning_insights()
        cost_projections = get_cost_projections()
        
        # Add trading mode info to response
        response_data = {
            'total_value': total_value,
            'balance': portfolio['balance'],
            'positions': portfolio['positions'],
            'trade_history': portfolio['trade_history'][-20:],
            'market_data': current_market_data,
            'learning_metrics': learning_insights,
            'cost_tracking': cost_projections,
            'timestamp': datetime.now().isoformat(),
            'trading_mode': 'REAL' if USE_REAL_TRADING else 'PAPER',
            'real_balance': get_real_balance() if USE_REAL_TRADING else None
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f"API Status Error: {e}")
        return jsonify({'error': str(e)}), 500


def run_flask_app():
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)

def get_batch_market_data():
    """Optimized batch market data fetching"""
    # Get all tickers in one API call (weight: 40)
    batch_url = "https://api.binance.com/api/v3/ticker/24hr"
    
    try:
        response = requests.get(batch_url)
        all_tickers = response.json()
        
        # Our target coins
        target_coins = ['BTC', 'ETH', 'SOL', 'BNB', 'ADA', 'DOGE', 
                       'AVAX', 'TAO', 'LINK', 'DOT', 'UNI', 'FET']
        
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
    
def kelly_size(win_rate, risk_reward):
    # b = risk/reward ratio, p = win rate, q = 1-p
    b = risk_reward
    p = win_rate
    q = 1 - p
    kelly = (b*p - q)/b if b > 0 else 0
    # Be conservative: clamp to [0, 0.25]
    return max(0, min(0.25, kelly))

def get_quick_prices(coins):
    """Get only prices for specific coins - lightweight"""
    prices = {}
    
    for coin in coins:
        try:
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={coin}USDT"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                prices[coin] = float(data['price'])
        except:
            pass
    
    return prices

def get_multi_timeframe_data(symbol):
    """Get multi-timeframe data for a single symbol"""
    base_url = "https://api.binance.com/api/v3/klines"
    intervals = {
        '5m': ('5m', 48),    # 4 hours of 5-minute candles
        '15m': ('15m', 48),  # 12 hours of 15-minute candles
        '1h': ('1h', 48),    # 48 hours of hourly candles
        '4h': ('4h', 42),    # 7 days of 4-hour candles
        '1d': ('1d', 30)     # 30 days of daily candles
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

def calculate_vwap(prices, volumes):
    """Calculate Volume Weighted Average Price"""
    if not prices or not volumes or len(prices) != len(volumes):
        return 0
    
    total_volume = sum(volumes)
    if total_volume == 0:
        return sum(prices) / len(prices)
    
    vwap = sum(p * v for p, v in zip(prices, volumes)) / total_volume
    return vwap

def identify_support_resistance(candles, lookback=20):
    """Identify key support and resistance levels"""
    if len(candles) < lookback:
        return {'support': [], 'resistance': []}
    
    highs = [float(c[2]) for c in candles[-lookback:]]
    lows = [float(c[3]) for c in candles[-lookback:]]
    closes = [float(c[4]) for c in candles[-lookback:]]
    
    # Find local peaks and troughs
    resistance_levels = []
    support_levels = []
    
    for i in range(2, len(highs) - 2):
        # Resistance: local high
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
           highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            resistance_levels.append(highs[i])
        
        # Support: local low
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
           lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            support_levels.append(lows[i])
    
    # Cluster nearby levels
    def cluster_levels(levels, threshold=0.01):
        if not levels:
            return []
        
        clustered = []
        levels.sort()
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if (level - current_cluster[-1]) / current_cluster[-1] < threshold:
                current_cluster.append(level)
            else:
                clustered.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [level]
        
        if current_cluster:
            clustered.append(sum(current_cluster) / len(current_cluster))
        
        return clustered
    
    return {
        'support': cluster_levels(support_levels),
        'resistance': cluster_levels(resistance_levels)
    }

def calculate_volume_profile(prices, volumes, bins=10):
    """Calculate volume profile to identify high-liquidity zones"""
    if not prices or not volumes:
        return {'poc': 0, 'val_high': 0, 'val_low': 0}
    
    min_price = min(prices)
    max_price = max(prices)
    
    if min_price == max_price:
        return {'poc': min_price, 'val_high': min_price, 'val_low': min_price}
    
    # Create price bins
    bin_size = (max_price - min_price) / bins
    volume_by_price = {}
    
    for price, volume in zip(prices, volumes):
        bin_index = min(int((price - min_price) / bin_size), bins - 1)
        bin_price = min_price + (bin_index + 0.5) * bin_size
        
        if bin_price not in volume_by_price:
            volume_by_price[bin_price] = 0
        volume_by_price[bin_price] += volume
    
    if not volume_by_price:
        return {'poc': 0, 'val_high': 0, 'val_low': 0}
    
    # Find Point of Control (highest volume price)
    poc = max(volume_by_price.keys(), key=lambda x: volume_by_price[x])
    
    # Find Value Area (70% of volume)
    total_volume = sum(volume_by_price.values())
    target_volume = total_volume * 0.7
    
    sorted_prices = sorted(volume_by_price.keys(), key=lambda x: volume_by_price[x], reverse=True)
    accumulated_volume = 0
    value_area_prices = []
    
    for price in sorted_prices:
        accumulated_volume += volume_by_price[price]
        value_area_prices.append(price)
        if accumulated_volume >= target_volume:
            break
    
    val_high = max(value_area_prices) if value_area_prices else poc
    val_low = min(value_area_prices) if value_area_prices else poc
    
    return {
        'poc': poc,
        'val_high': val_high,
        'val_low': val_low,
        'volume_nodes': volume_by_price
    }

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
    volumes = hourly_data['volumes']
    
    indicators = {}
    
    # 1. ADX for trend strength (simplified version)
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
    
    # 2. Bollinger Bands width for volatility
    sma_20 = sum(closes[-20:]) / 20
    std_dev = math.sqrt(sum((c - sma_20) ** 2 for c in closes[-20:]) / 20)
    bb_width = (std_dev * 2) / sma_20 * 100  # As percentage
    indicators['bb_width'] = bb_width
    
    # 3. Price position relative to moving averages
    sma_50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else sma_20
    ema_9 = calculate_ema(closes, 9)
    
    price_above_sma = current_price > sma_20
    price_above_sma50 = current_price > sma_50
    indicators['price_above_sma'] = price_above_sma
    
    # 4. Higher highs/lower lows for trend detection
    recent_highs = highs[-10:]
    recent_lows = lows[-10:]
    
    higher_highs = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i] > recent_highs[i-1]) > 6
    lower_lows = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] < recent_lows[i-1]) > 6
    
    # 5. Volume trend
    avg_volume_recent = sum(volumes[-5:]) / 5 if len(volumes) >= 5 else 0
    avg_volume_older = sum(volumes[-20:-10]) / 10 if len(volumes) >= 20 else avg_volume_recent
    volume_increasing = avg_volume_recent > avg_volume_older * 1.2
    indicators['volume_trend'] = 'increasing' if volume_increasing else 'stable'
    
    # Determine regime
    confidence = 0
    regime = 'UNKNOWN'
    
    if adx > 25:  # Strong trend
        if higher_highs or (price_above_sma and price_above_sma50):
            regime = 'TRENDING_UP'
            confidence = min(adx / 40 * 100, 90)
        elif lower_lows or (not price_above_sma and not price_above_sma50):
            regime = 'TRENDING_DOWN'
            confidence = min(adx / 40 * 100, 90)
    elif adx < 20 and bb_width < 3:  # Ranging market
        regime = 'RANGING'
        confidence = 70
    elif bb_width > 5:  # High volatility
        regime = 'VOLATILE'
        confidence = min(bb_width * 10, 80)
    else:
        regime = 'TRANSITIONAL'
        confidence = 50
    
    return {
        'regime': regime,
        'confidence': confidence,
        'indicators': indicators
    }

def calculate_ema(prices, period):
    """Calculate Exponential Moving Average"""
    if len(prices) < period:
        return 0
    
    multiplier = 2 / (period + 1)
    ema = sum(prices[:period]) / period  # Initial SMA
    
    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema
    
    return ema

def calculate_macd(prices):
    """Calculate MACD indicator"""
    if len(prices) < 26:
        return {'macd': 0, 'signal': 0, 'histogram': 0}
    
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    macd_line = ema_12 - ema_26
    
    # For signal line, we need MACD history
    macd_values = []
    for i in range(26, len(prices)):
        subset = prices[:i+1]
        ema_12_temp = calculate_ema(subset, 12)
        ema_26_temp = calculate_ema(subset, 26)
        macd_values.append(ema_12_temp - ema_26_temp)
    
    signal_line = calculate_ema(macd_values, 9) if len(macd_values) >= 9 else 0
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }

def calculate_dynamic_stops(current_price, atr, support_levels, resistance_levels, direction, market_regime):
    """Calculate dynamic stop loss and take profit based on market conditions"""
    
    sl_multiplier = get_hyperparameter('sl_multiplier', 1.5)
    tp_multiplier = get_hyperparameter('tp_multiplier', 2.5)
    
    if market_regime == 'VOLATILE':
        sl_multiplier = 2.0  # Wider stop in volatile markets
        tp_multiplier = 3.5  # Larger targets
    elif market_regime == 'RANGING':
        sl_multiplier = 1.0  # Tighter stop in ranging markets
        tp_multiplier = 1.5  # Smaller targets
    elif market_regime in ['TRENDING_UP', 'TRENDING_DOWN']:
        sl_multiplier = 1.8  # Give trend room to breathe
        tp_multiplier = 4.0  # Larger targets in trends
    
    # Calculate ATR-based stops
    atr_stop_distance = atr * sl_multiplier
    atr_target_distance = atr * tp_multiplier
    
    if direction == 'LONG':
        # Initial ATR-based levels
        stop_loss = current_price - atr_stop_distance
        take_profit = current_price + atr_target_distance
        
        # Adjust stop loss to nearest support
        if support_levels:
            nearest_support = max([s for s in support_levels if s < current_price], default=None)
            if nearest_support and nearest_support > stop_loss:
                stop_loss = nearest_support * 0.998  # Just below support
        
        # Adjust take profit to nearest resistance
        if resistance_levels:
            nearest_resistance = min([r for r in resistance_levels if r > current_price], default=None)
            if nearest_resistance and nearest_resistance < take_profit:
                # Only adjust if resistance is at least 1.5x ATR away
                if (nearest_resistance - current_price) > atr * 1.5:
                    take_profit = nearest_resistance * 0.998  # Just below resistance
    
    else:  # SHORT
        # Initial ATR-based levels
        stop_loss = current_price + atr_stop_distance
        take_profit = current_price - atr_target_distance
        
        # Adjust stop loss to nearest resistance
        if resistance_levels:
            nearest_resistance = min([r for r in resistance_levels if r > current_price], default=None)
            if nearest_resistance and nearest_resistance < stop_loss:
                stop_loss = nearest_resistance * 1.002  # Just above resistance
        
        # Adjust take profit to nearest support
        if support_levels:
            nearest_support = max([s for s in support_levels if s < current_price], default=None)
            if nearest_support and nearest_support > take_profit:
                # Only adjust if support is at least 1.5x ATR away
                if (current_price - nearest_support) > atr * 1.5:
                    take_profit = nearest_support * 1.002  # Just above support
    
    # Calculate percentages for validation
    sl_percentage = abs(stop_loss - current_price) / current_price
    tp_percentage = abs(take_profit - current_price) / current_price
    
    # Ensure minimum and maximum boundaries
    min_sl = 0.01  # 1% minimum
    max_sl = 0.05  # 5% maximum
    min_tp = 0.015  # 1.5% minimum
    max_tp = 0.10  # 10% maximum
    
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
            
            # Multiple indicators per timeframe
            sma_short = sum(closes[-5:]) / 5
            sma_mid = sum(closes[-10:]) / 10
            sma_long = sum(closes[-20:]) / 20
            
            # RSI
            rsi = calculate_rsi(closes)
            
            # MACD
            macd_data = calculate_macd(closes)
            
            # Scoring for this timeframe
            tf_score = 0
            
            # Trend alignment
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
            
            # RSI confirmation
            if rsi > 60 and tf_score > 0:
                tf_score += 0.5
            elif rsi < 40 and tf_score < 0:
                tf_score -= 0.5
            
            # MACD confirmation
            if macd_data['histogram'] > 0 and tf_score > 0:
                tf_score += 0.5
            elif macd_data['histogram'] < 0 and tf_score < 0:
                tf_score -= 0.5
            
            alignment_score += tf_score
            strength_scores.append(abs(tf_score))
    
    # Calculate overall strength
    avg_strength = sum(strength_scores) / len(strength_scores) if strength_scores else 0
    
    # Determine overall direction with higher thresholds
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

def quick_position_check(position_id, position, current_price):
    """Lightweight position check for SL/TP only"""
    coin = position['coin']
    entry_price = position['entry_price']
    direction = position['direction']
    
    should_close = False
    close_reason = ""
    
    # Calculate P&L
    if direction == 'LONG':
        pnl_percent = (current_price - entry_price) / entry_price
        if current_price <= position['stop_loss']:
            should_close = True
            close_reason = "Stop Loss Hit"
        elif current_price >= position['take_profit']:
            should_close = True
            close_reason = "Take Profit Hit"
    else:  # SHORT
        pnl_percent = (entry_price - current_price) / entry_price
        if current_price >= position['stop_loss']:
            should_close = True
            close_reason = "Stop Loss Hit"
        elif current_price <= position['take_profit']:
            should_close = True
            close_reason = "Take Profit Hit"
    
    pnl_amount = pnl_percent * position['notional_value']
    
    if should_close:
        close_position(position_id, position, current_price, close_reason, pnl_amount)
    
    return should_close

def smart_position_analysis(position_id, position, market_data, sentiment, multi_tf_data):
    """Enhanced position analysis with trailing stops and dynamic adjustments"""
    coin = position['coin']
    current_price = market_data[coin]['price']
    entry_price = position['entry_price']
    direction = position['direction']
    
    # Calculate P&L
    if direction == 'LONG':
        pnl_percent = (current_price - entry_price) / entry_price
    else:
        pnl_percent = (entry_price - current_price) / entry_price
    
    pnl_amount = pnl_percent * position['notional_value']
    
    # First check standard SL/TP
    should_close = False
    close_reason = ""
    
    if direction == 'LONG':
        if current_price <= position['stop_loss']:
            should_close = True
            close_reason = "Stop Loss Hit"
        elif current_price >= position['take_profit']:
            should_close = True
            close_reason = "Take Profit Hit"
    else:
        if current_price >= position['stop_loss']:
            should_close = True
            close_reason = "Stop Loss Hit"
        elif current_price <= position['take_profit']:
            should_close = True
            close_reason = "Take Profit Hit"
    
    # Trailing stop implementation
    if not should_close and pnl_percent > 0.02:  # If in profit > 2%
        if multi_tf_data and '1h' in multi_tf_data:
            hourly_data = multi_tf_data['1h']
            if 'highs' in hourly_data and 'lows' in hourly_data and 'closes' in hourly_data:
                atr = calculate_atr(hourly_data['highs'], hourly_data['lows'], hourly_data['closes'])
                
                if atr > 0:
                    trailing_distance = atr * 1.5
                    
                    if direction == 'LONG':
                        new_stop = current_price - trailing_distance
                        if new_stop > position['stop_loss'] and new_stop > entry_price:
                            # Update stop loss in position
                            position['stop_loss'] = new_stop
                            print(f"   📍 Trailing stop updated for {coin}: ${new_stop:.2f}")
                    else:  # SHORT
                        new_stop = current_price + trailing_distance
                        if new_stop < position['stop_loss'] and new_stop < entry_price:
                            position['stop_loss'] = new_stop
                            print(f"   📍 Trailing stop updated for {coin}: ${new_stop:.2f}")
    
    # Smart exit analysis only if not hitting SL/TP and significant sentiment shift
    if not should_close and abs(sentiment) > 50:
        # Detect market regime change
        if multi_tf_data:
            market_regime = detect_market_regime(multi_tf_data, current_price)
            
            # Exit if regime has changed dramatically
            if position.get('market_regime') and market_regime['regime'] != position.get('market_regime'):
                if (direction == 'LONG' and market_regime['regime'] == 'TRENDING_DOWN') or \
                   (direction == 'SHORT' and market_regime['regime'] == 'TRENDING_UP'):
                    if pnl_percent > -0.01:  # Only if loss less than 1%
                        should_close = True
                        close_reason = f"Regime Change: {market_regime['regime']}"
        
        # AI-based exit for extreme sentiment shifts
        if not should_close and ((direction == 'LONG' and sentiment < -60) or \
                                 (direction == 'SHORT' and sentiment > 60)):
            
            if client and pnl_percent > -0.015:  # Only if loss less than 1.5%
                try:
                    exit_prompt = f"""
QUICK EXIT DECISION for {direction} {coin}:
Entry: ${entry_price:.2f}, Current: ${current_price:.2f}
P&L: {pnl_percent*100:.1f}%
Sentiment shift: {sentiment:.0f}

Should we exit? Reply: CLOSE or HOLD (one word)
"""
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": exit_prompt}],
                        max_tokens=10,
                        temperature=0.3
                    )
                    
                    # Track minimal cost
                    track_cost('openai', 0.00001, '50')
                    
                    if "CLOSE" in response.choices[0].message.content.upper():
                        should_close = True
                        close_reason = f"Smart Exit: Sentiment {sentiment:.0f}"
                        
                except Exception as e:
                    print(f"Exit analysis error: {e}")
    
    if should_close:
        close_position(position_id, position, current_price, close_reason, pnl_amount)
    
    return pnl_amount

def ai_trade_decision(coin, market_data, multi_tf_data, learning_insights):
    """Enhanced AI trading decision with advanced indicators"""
    if not client:
        return None
    
    current_price = market_data[coin]['price']
    tf_alignment = analyze_timeframe_alignment(multi_tf_data)
    
    # Skip if timeframes not aligned
    if not tf_alignment['aligned']:
        return {'direction': 'SKIP', 'reason': 'Timeframes not aligned'}
    
    # Calculate advanced indicators
    hourly_data = multi_tf_data.get('1h', {})
    daily_data = multi_tf_data.get('1d', {})
    
    # ATR for volatility
    atr = 0
    if 'highs' in hourly_data and 'lows' in hourly_data and 'closes' in hourly_data:
        atr = calculate_atr(hourly_data['highs'], hourly_data['lows'], hourly_data['closes'])
    
    # Support/Resistance levels
    sr_levels = identify_support_resistance(hourly_data.get('candles', []))
    
    # Volume Profile
    volume_profile = calculate_volume_profile(
        hourly_data.get('closes', []), 
        hourly_data.get('volumes', [])
    )
    
    # Market Regime
    market_regime = detect_market_regime(multi_tf_data, current_price)
    
    # VWAP
    vwap = calculate_vwap(hourly_data.get('closes', []), hourly_data.get('volumes', []))
    
    # Dynamic stops calculation
    dynamic_stops = calculate_dynamic_stops(
        current_price, 
        atr, 
        sr_levels.get('support', []), 
        sr_levels.get('resistance', []),
        'LONG' if tf_alignment['direction'] in ['bullish', 'strong_bullish'] else 'SHORT',
        market_regime['regime']
    )
    
    # Enhanced prompt with all indicators
    analysis_prompt = f"""
{coin} ADVANCED TRADING ANALYSIS

Price: ${current_price:.2f}
24h: {market_data[coin]['change_24h']:+.1f}%
Volume: {market_data[coin]['volume']/1000000:.1f}M

MARKET STRUCTURE:
- Regime: {market_regime['regime']} (confidence: {market_regime['confidence']:.0f}%)
- Timeframe Alignment: {tf_alignment['direction']} (score: {tf_alignment['score']:.1f})
- ATR: ${atr:.2f}
- VWAP: ${vwap:.2f}

LEVELS:
- POC (High Volume): ${volume_profile['poc']:.2f}
- Value Area: ${volume_profile['val_low']:.2f} - ${volume_profile['val_high']:.2f}
- Support: {', '.join([f'${s:.2f}' for s in sr_levels['support'][:3]])}
- Resistance: {', '.join([f'${r:.2f}' for r in sr_levels['resistance'][:3]])}

DYNAMIC RISK:
- Suggested SL: {dynamic_stops['sl_percentage']:.1f}%
- Suggested TP: {dynamic_stops['tp_percentage']:.1f}%
- Risk/Reward: {dynamic_stops['risk_reward']:.1f}

LEARNING:
- Win Rate: {learning_insights.get('win_rate', 0):.0f}%
- Best Leverage: {learning_insights.get('best_leverage', 10)}x

LEVERAGE GUIDELINES:
- 9-10 confidence: Use 20x leverage
- 8 confidence: Use 15x leverage  
- 7 confidence: Use 12x leverage
- Below 7: Use 5-10x leverage

Based on market structure, volume profile, and regime, provide:
DECISION: [LONG/SHORT/SKIP]
LEVERAGE: [5-20]
SIZE: [3-10]%
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
        
        # Track cost
        track_cost('openai', 0.00008, '300')
        
        # Parse response
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
                    trade_params['leverage'] = min(20, max(5, int(''.join(filter(str.isdigit, value)))))
                elif 'SIZE' in key:
                    size = float(''.join(filter(lambda x: x.isdigit() or x == '.', value)))
                    trade_params['position_size'] = min(0.10, max(0.03, size / 100))
                elif 'CONFIDENCE' in key:
                    trade_params['confidence'] = min(10, max(1, int(''.join(filter(str.isdigit, value)))))
                elif 'REASON' in key:
                    trade_params['reasoning'] = value[:150]
        
        # 🔥 NEW: CONFIDENCE-BASED LEVERAGE OVERRIDE 🔥
        if 'confidence' in trade_params:
            if trade_params['confidence'] >= 9:
                trade_params['leverage'] = 20
            elif trade_params['confidence'] >= 8:
                trade_params['leverage'] = 15
            elif trade_params['confidence'] >= 7:
                trade_params['leverage'] = 12
            else:
                trade_params['leverage'] = max(5, trade_params.get('leverage', 10))
        
        # Use dynamic stops
        trade_params['stop_loss'] = dynamic_stops['stop_loss']
        trade_params['take_profit'] = dynamic_stops['take_profit']
        trade_params['sl_percentage'] = dynamic_stops['sl_percentage']
        trade_params['tp_percentage'] = dynamic_stops['tp_percentage']
        trade_params['market_regime'] = market_regime['regime']
        trade_params['atr'] = atr
        
        # Default values if parsing fails
        if 'direction' not in trade_params:
            trade_params['direction'] = 'SKIP'
        if 'leverage' not in trade_params:
            trade_params['leverage'] = 10
        if 'position_size' not in trade_params:
            trade_params['position_size'] = 0.05
        if 'confidence' not in trade_params:
            trade_params['confidence'] = 5
        if 'reasoning' not in trade_params:
            trade_params['reasoning'] = f"{market_regime['regime']} market, {tf_alignment['direction']} alignment"
        
        trade_params['duration'] = 'SWING'  # Default
        
        return trade_params
        
    except Exception as e:
        print(f"AI Error: {e}")
        return None

def execute_trade(coin, trade_params, current_price):
    """Execute trade with enhanced position management + REAL TRADING"""
    global portfolio
    
    if trade_params['direction'] == 'SKIP':
        return
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Get real balance if using real trading
    if USE_REAL_TRADING:
        real_balance = get_real_balance()
        if real_balance != portfolio['balance']:
            portfolio['balance'] = real_balance
            print(f"   💰 Real Balance Updated: ${real_balance:.2f}")
    
    # Position sizing (UNCHANGED)
    winrate, rr = get_recent_winrate_and_rr(50)
    kelly_fraction = kelly_size(winrate, rr)
    bet_fraction = kelly_fraction * 0.5
    position_value = portfolio['balance'] * bet_fraction
    if position_value < portfolio['balance'] * 0.02:
        position_value = portfolio['balance'] * trade_params.get('position_size', 0.05)
    
    leverage = trade_params['leverage']
    notional_value = position_value * leverage
    
    # Calculate quantity for Binance
    quantity = notional_value / current_price
    
    # Use dynamic stops (UNCHANGED)
    stop_loss_price = trade_params['stop_loss']
    take_profit_price = trade_params['take_profit']
    
    position_id = f"{coin}_{timestamp.replace(' ', '_').replace(':', '-')}"
    
    # REAL TRADING: Place the order
    if USE_REAL_TRADING:
        symbol = f"{coin}USDT"
        side = 'BUY' if trade_params['direction'] == 'LONG' else 'SELL'
        
        order_result = place_real_order(symbol, side, quantity, leverage)
        
        if not order_result:
            print(f"❌ REAL ORDER FAILED for {coin}")
            return
        else:
            print(f"✅ REAL ORDER EXECUTED: {order_result.get('orderId')}")
    
    # Store position data (UNCHANGED)
    position = {
        'coin': coin,
        'direction': trade_params['direction'],
        'entry_price': current_price,
        'entry_time': timestamp,
        'position_size': position_value,
        'leverage': leverage,
        'notional_value': notional_value,
        'stop_loss': stop_loss_price,
        'take_profit': take_profit_price,
        'duration_target': trade_params.get('duration', 'SWING'),
        'confidence': trade_params['confidence'],
        'reasoning': trade_params['reasoning'],
        'market_regime': trade_params.get('market_regime', 'UNKNOWN'),
        'atr': trade_params.get('atr', 0),
        'quantity': quantity  # Add this for real trading
    }
    
    portfolio['positions'][position_id] = position
    portfolio['balance'] -= position_value
    
    # Save to database (UNCHANGED)
    save_active_position(position_id, position)
    save_trade_to_db({
        'time': timestamp,
        'position_id': position_id,
        'coin': coin,
        'action': f"{trade_params['direction']} OPEN",
        'direction': trade_params['direction'],
        'price': current_price,
        'position_size': position_value,
        'leverage': leverage,
        'notional_value': notional_value,
        'stop_loss': stop_loss_price,
        'take_profit': take_profit_price,
        'confidence': trade_params['confidence'],
        'reason': trade_params['reasoning']
    })
    
    # Display (UNCHANGED)
    emoji = "📈" if trade_params['direction'] == 'LONG' else "📉"
    risk_reward = trade_params.get('tp_percentage', 0) / trade_params.get('sl_percentage', 1) if trade_params.get('sl_percentage', 0) > 0 else 0
    
    trading_mode = "🔴 REAL" if USE_REAL_TRADING else "📝 PAPER"
    
    print(f"\n   {emoji} {trade_params['direction']} {coin}: ${position_value:.2f} @ {leverage}x {trading_mode}")
    print(f"      Market: {trade_params.get('market_regime', 'UNKNOWN')} | Confidence: {trade_params['confidence']}/10")
    print(f"      SL: ${stop_loss_price:.2f} (-{trade_params.get('sl_percentage', 0):.1f}%) | TP: ${take_profit_price:.2f} (+{trade_params.get('tp_percentage', 0):.1f}%)")
    print(f"      Risk/Reward: 1:{risk_reward:.1f} | Reason: {trade_params['reasoning'][:50]}...")

def close_position(position_id, position, current_price, reason, pnl_amount):
    """Close position + REAL TRADING"""
    global portfolio
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    pnl_percent = (pnl_amount / position['position_size']) * 100
    
    # REAL TRADING: Close the position
    if USE_REAL_TRADING:
        symbol = f"{position['coin']}USDT"
        # Opposite side to close
        side = 'SELL' if position['direction'] == 'LONG' else 'BUY'
        quantity = position.get('quantity', position['notional_value'] / current_price)
        
        close_result = close_real_position(symbol, side, quantity)
        
        if close_result:
            print(f"✅ REAL POSITION CLOSED: {close_result.get('orderId')}")
        else:
            print(f"❌ REAL CLOSE FAILED for {position['coin']}")
    
    # Return funds (UNCHANGED)
    final_amount = position['position_size'] + pnl_amount
    portfolio['balance'] += max(0, final_amount)
    
    # Save closing trade (UNCHANGED)
    save_trade_to_db({
        'time': timestamp,
        'position_id': position_id,
        'coin': position['coin'],
        'action': f"{position['direction']} CLOSE",
        'direction': position['direction'],
        'price': current_price,
        'position_size': position['position_size'],
        'pnl': pnl_amount,
        'pnl_percent': pnl_percent,
        'reason': reason,
        'duration': str(datetime.now() - datetime.strptime(position['entry_time'], '%Y-%m-%d %H:%M:%S'))
    })
    
    # Add to history (UNCHANGED)
    portfolio['trade_history'].append({
        'time': timestamp,
        'coin': position['coin'],
        'pnl': pnl_amount,
        'pnl_percent': pnl_percent
    })
    
    # Remove position (UNCHANGED)
    del portfolio['positions'][position_id]
    remove_active_position(position_id)
    
    emoji = "💰" if pnl_amount > 0 else "💔"
    trading_mode = "🔴 REAL" if USE_REAL_TRADING else "📝 PAPER"
    
    print(f"   {emoji} CLOSED {position['coin']}: ${pnl_amount:+.2f} ({pnl_percent:+.1f}%) - {reason} {trading_mode}")

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
    
    # Win rate
    cursor.execute('SELECT COUNT(*) FROM trades WHERE pnl IS NOT NULL')
    total = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM trades WHERE pnl > 0')
    wins = cursor.fetchone()[0]
    
    insights['win_rate'] = (wins / total * 100) if total > 0 else 0
    insights['total_trades'] = total
    
    # Best leverage
    cursor.execute('''
        SELECT leverage, AVG(pnl_percent) as avg_return
        FROM trades 
        WHERE pnl IS NOT NULL 
        GROUP BY leverage 
        ORDER BY avg_return DESC 
        LIMIT 1
    ''')
    
    best_lev = cursor.fetchone()
    insights['best_leverage'] = best_lev[0] if best_lev else 10
    
    # Average P&L
    cursor.execute('SELECT AVG(pnl) FROM trades WHERE pnl > 0')
    avg_profit = cursor.fetchone()[0]
    insights['avg_profit'] = avg_profit if avg_profit else 0
    
    cursor.execute('SELECT AVG(pnl) FROM trades WHERE pnl < 0')
    avg_loss = cursor.fetchone()[0]
    insights['avg_loss'] = avg_loss if avg_loss else 0
    
    # Market regime performance
    cursor.execute('''
        SELECT market_conditions, COUNT(*) as count, AVG(pnl_percent) as avg_pnl
        FROM trades
        WHERE market_conditions IS NOT NULL AND pnl IS NOT NULL
        GROUP BY market_conditions
    ''')
    
    regime_performance = cursor.fetchall()
    insights['regime_performance'] = {row[0]: {'count': row[1], 'avg_pnl': row[2]} for row in regime_performance}
    
    conn.close()
    return insights

def get_recent_winrate_and_rr(window=50):
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    cursor.execute('SELECT pnl, pnl_percent, stop_loss, take_profit FROM trades WHERE pnl IS NOT NULL AND stop_loss IS NOT NULL AND take_profit IS NOT NULL ORDER BY timestamp DESC LIMIT ?', (window,))
    trades = cursor.fetchall()
    conn.close()
    if not trades or len(trades) < 5:
        return 0.5, 2.0  # Default: 50% win, 2R
    wins = [t for t in trades if t[0] > 0]
    winrate = len(wins)/len(trades)
    rrs = [abs((t[3]-t[2])/(t[2]-t[3])) if (t[3]!=t[2]) else 2.0 for t in trades]
    avg_rr = sum(rrs)/len(rrs)
    return winrate, avg_rr

def auto_tune_hyperparameters(window=50):
    """Auto-tune ATR multipliers and confidence threshold using last N trades"""
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    # Get last N trades
    cursor.execute('''
        SELECT stop_loss, take_profit, pnl_percent, confidence
        FROM trades
        WHERE stop_loss IS NOT NULL AND take_profit IS NOT NULL AND pnl_percent IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (window,))
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        return  # Not enough data

    # Try different ATR multipliers and confidence levels, find best Sharpe ratio
    best_score = -9999
    best_params = {}
    for sl_mult in [1.0, 1.2, 1.5, 1.8, 2.0]:
        for tp_mult in [1.5, 2.0, 2.5, 3.0, 4.0]:
            for conf in [5, 6, 7, 8]:
                # Simulate "filter" over history
                filtered = [row for row in rows if row[3] >= conf]
                if len(filtered) < 10: continue
                rr = [(row[1]-row[0])/abs(row[0]-row[1]) for row in filtered]  # simple R/R check
                profits = [row[2] for row in filtered]
                if len(profits) < 2: continue
                mean = sum(profits)/len(profits)
                std = statistics.stdev(profits)
                score = mean/std if std > 0 else mean
                # Score must penalize too few trades:
                score -= 0.05 * (20 - len(filtered)) if len(filtered) < 20 else 0
                if score > best_score:
                    best_score = score
                    best_params = {'sl_mult': sl_mult, 'tp_mult': tp_mult, 'conf': conf}

    # Save best found params to database
    if best_params:
        set_hyperparameter('sl_multiplier', best_params['sl_mult'])
        set_hyperparameter('tp_multiplier', best_params['tp_mult'])
        set_hyperparameter('min_confidence', best_params['conf'])
        print(f"\n🔧 Auto-tuned params: SL x{best_params['sl_mult']} | TP x{best_params['tp_mult']} | MinConf {best_params['conf']} | Sharpe {best_score:.2f}")

def calculate_portfolio_value(market_data):
    """Calculate total portfolio value"""
    total = portfolio['balance']
    
    for pos_id, position in portfolio['positions'].items():
        coin = position['coin']
        if coin in market_data:
            current_price = market_data[coin]['price']
            entry_price = position['entry_price']
            
            if position['direction'] == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price
            
            pnl = pnl_pct * position['notional_value']
            total += position['position_size'] + pnl
    
    return max(0, total)

def run_enhanced_bot():
    """Main bot loop with enhanced features"""
    # Initialize
    init_database()
    portfolio['positions'] = load_active_positions()
    sync_real_balance()
    
    print("🚀 ENHANCED SMART TRADING BOT - ADVANCED MODE")
    print("⚡ Quick checks: Every 15 seconds (positions + trailing stops)")
    print("🧠 Full analysis: Every 2 minutes (complete scan)")
    print("📊 Advanced Features:")
    print("   • Volume Profile & Liquidity Analysis")
    print("   • Support/Resistance Detection")
    print("   • Market Regime Detection (Trending/Ranging/Volatile)")
    print("   • Dynamic Stop Loss & Take Profit")
    print("   • Trailing Stops for Winners")
    print("   • Multi-Indicator Confluence")
    print("💰 Max positions: 4 concurrent")
    print("🎯 Min confidence: 6/10")
    print("📊 Coins: BTC, ETH, SOL, BNB, ADA, DOGE, AVAX, TAO, LINK, DOT, UNI, FET")
    print("🌐 Dashboard: http://localhost:5000")
    print("="*80)
    
    # Price histories for technical analysis
    target_coins = ['BTC', 'ETH', 'SOL', 'BNB', 'ADA', 'DOGE', 
                   'AVAX', 'TAO', 'LINK', 'DOT', 'UNI', 'FET']
    price_histories = {coin: [] for coin in target_coins}
    
    # Timeframe data storage for pattern recognition
    timeframe_data = {coin: {} for coin in target_coins}
    
    # Pattern memory for better decisions
    pattern_memory = {}
    
    last_full_analysis = 0
    last_balance_sync = 0
    iteration = 0
    quick_checks_count = 0
    full_analyses_count = 0
    
    while True:
        try:
            iteration += 1
            current_time = time.time()

            if current_time - last_balance_sync >= 300:  # 5 minutes
                if USE_REAL_TRADING:
                    sync_real_balance()
                last_balance_sync = current_time
            
            # ========== QUICK CHECK MODE (Every 15 seconds) ==========
            if portfolio['positions']:
                # Get only prices for active positions (minimal API weight)
                active_coins = list(set(p['coin'] for p in portfolio['positions'].values()))
                quick_prices = get_quick_prices(active_coins)
                
                positions_closed = 0
                for pos_id, position in list(portfolio['positions'].items()):
                    coin = position['coin']
                    if coin in quick_prices:
                        was_closed = quick_position_check(pos_id, position, quick_prices[coin])
                        if was_closed:
                            positions_closed += 1
                
                quick_checks_count += 1
                
                # Only print if not doing full analysis
                if current_time - last_full_analysis < FULL_ANALYSIS_INTERVAL:
                    status = f"⚡ Quick #{quick_checks_count}: {len(portfolio['positions'])} positions"
                    if positions_closed > 0:
                        status += f" | {positions_closed} closed"
                    print(f"\r{status}", end="", flush=True)
            
            # ========== FULL ANALYSIS MODE (Every 2 minutes) ==========
            if current_time - last_full_analysis >= FULL_ANALYSIS_INTERVAL:
                full_analyses_count += 1
                print(f"\n\n{'='*80}")
                print(f"🧠 FULL MARKET ANALYSIS #{full_analyses_count}")
                print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                mode_indicator = "🔴 REAL TRADING" if USE_REAL_TRADING else "📝 PAPER TRADING"
                print(f"Mode: {mode_indicator}")
                print(f"Current Balance: ${portfolio['balance']:.2f}")
                print(f"Quick checks since last analysis: {quick_checks_count}")
                print("="*80)
                
                quick_checks_count = 0  # Reset counter
                
                # Get comprehensive market data (batch call)
                market_data = get_batch_market_data()
                
                # Update price histories for all coins
                for coin in target_coins:
                    if coin in market_data:
                        price_histories[coin].append(market_data[coin]['price'])
                        if len(price_histories[coin]) > 100:  # Keep more history for better analysis
                            price_histories[coin] = price_histories[coin][-100:]
                
                # Display enhanced market overview
                print("\n📊 Market Overview with Advanced Metrics:")
                for coin in target_coins[:6]:  # Show first 6 coins
                    if coin in market_data:
                        data = market_data[coin]
                        
                        # Get multi-timeframe data for this coin
                        symbol = f"{coin}USDT"
                        coin_tf_data = get_multi_timeframe_data(symbol)
                        
                        # Calculate indicators
                        rsi = calculate_rsi(price_histories[coin]) if len(price_histories[coin]) > 14 else 50
                        
                        # Detect market regime
                        regime_data = detect_market_regime(coin_tf_data, data['price'])
                        regime = regime_data['regime']
                        
                        # Trend emoji based on regime
                        if regime == 'TRENDING_UP':
                            trend = "🚀"
                        elif regime == 'TRENDING_DOWN':
                            trend = "📉"
                        elif regime == 'RANGING':
                            trend = "↔️"
                        elif regime == 'VOLATILE':
                            trend = "⚡"
                        else:
                            trend = "❓"
                        
                        # Volume analysis
                        vol_ratio = data['volume'] / (sum([m.get('volume', 0) for m in market_data.values()]) / len(market_data))
                        vol_indicator = "🔥" if vol_ratio > 2 else "📊" if vol_ratio > 1 else "💤"
                        
                        print(f"   {coin}: ${data['price']:.2f} {trend} ({data['change_24h']:+.1f}%) RSI:{rsi:.0f} {vol_indicator} [{regime}]")
                
                # Advanced position management with enhanced analysis
                if portfolio['positions']:
                    print(f"\n📈 Advanced Position Management:")
                    sentiment_scores = {}
                    
                    # Calculate sentiment for all coins with more factors
                    for coin in market_data:
                        rsi = calculate_rsi(price_histories[coin]) if len(price_histories[coin]) > 14 else 50
                        momentum = market_data[coin]['change_24h']
                        volume_change = market_data[coin]['volume'] / market_data[coin].get('quote_volume', 1)
                        
                        # Complex sentiment calculation
                        sentiment = 0
                        
                        # RSI component
                        if rsi < 30:
                            sentiment += 40  # Oversold
                        elif rsi > 70:
                            sentiment -= 40  # Overbought
                        else:
                            sentiment += (50 - rsi) * 0.5
                        
                        # Momentum component
                        sentiment += min(max(momentum * 2, -30), 30)
                        
                        # Volume component
                        if volume_change > 1.5:
                            sentiment += 10 if momentum > 0 else -10
                        
                        sentiment_scores[coin] = sentiment
                    
                    # Analyze each position with full context
                    for pos_id, position in list(portfolio['positions'].items()):
                        coin = position['coin']
                        if coin in market_data:
                            # Get fresh multi-timeframe data for position
                            symbol = f"{coin}USDT"
                            position_tf_data = get_multi_timeframe_data(symbol)
                            
                            current_price = market_data[coin]['price']
                            entry_price = position['entry_price']
                            
                            # Calculate current P&L
                            if position['direction'] == 'LONG':
                                pnl_pct = (current_price - entry_price) / entry_price
                            else:
                                pnl_pct = (entry_price - current_price) / entry_price
                            
                            pnl_amount = pnl_pct * position['notional_value']
                            
                            # Calculate distance to stops
                            sl_distance = abs(current_price - position['stop_loss']) / current_price * 100
                            tp_distance = abs(position['take_profit'] - current_price) / current_price * 100
                            
                            print(f"   • {position['direction']} {coin}: P&L ${pnl_amount:+.2f} ({pnl_pct*100:+.1f}%)")
                            print(f"     SL: -{sl_distance:.1f}% | TP: +{tp_distance:.1f}% | Sentiment: {sentiment_scores.get(coin, 0):.0f}")
                            
                            # Smart position analysis with enhanced features
                            smart_position_analysis(pos_id, position, market_data, sentiment_scores.get(coin, 0), position_tf_data)
                
                # Look for new trading opportunities with enhanced analysis
                if len(portfolio['positions']) < MAX_CONCURRENT_POSITIONS:
                    if all(len(history) >= 20 for history in price_histories.values()):
                        learning_insights = get_learning_insights()
                        
                        print(f"\n🤖 AI Advanced Opportunity Scan:")
                        print(f"   Historical Win Rate: {learning_insights['win_rate']:.1f}%")
                        print(f"   Optimal Leverage: {learning_insights.get('best_leverage', 10)}x")
                        
                        # Show regime performance if available
                        if 'regime_performance' in learning_insights and learning_insights['regime_performance']:
                            print(f"   Best Market Regime: ", end="")
                            best_regime = max(learning_insights['regime_performance'].items(), 
                                            key=lambda x: x[1].get('avg_pnl', 0))
                            print(f"{best_regime[0]} ({best_regime[1]['avg_pnl']:.1f}% avg)")
                        
                        print(f"   Scanning {len(target_coins)} coins with advanced indicators...")
                        
                        opportunities_found = 0
                        opportunities_analyzed = 0
                        
                        for coin in target_coins:
                            # Skip if we already have a position
                            if any(p['coin'] == coin for p in portfolio['positions'].values()):
                                continue
                            
                            # Skip if balance too low
                            if portfolio['balance'] < 50:
                                print(f"   ⚠️ Insufficient balance (${portfolio['balance']:.2f})")
                                break
                            
                            opportunities_analyzed += 1
                            
                            # Get multi-timeframe data for this coin
                            symbol = f"{coin}USDT"
                            multi_tf_data = get_multi_timeframe_data(symbol)
                            
                            # Store timeframe data for pattern recognition
                            timeframe_data[coin] = multi_tf_data
                            
                            # Check market regime first
                            regime_data = detect_market_regime(multi_tf_data, market_data[coin]['price'])
                            
                            # Skip if market is too volatile or transitional
                            if regime_data['regime'] == 'TRANSITIONAL' and regime_data['confidence'] < 60:
                                continue
                            
                            # AI trade decision with all context
                            trade_params = ai_trade_decision(coin, market_data, multi_tf_data, learning_insights)
                            
                            if trade_params and trade_params['direction'] != 'SKIP':
                                if trade_params['confidence'] >= get_hyperparameter('min_confidence', 6):
                                    # Additional risk/reward check
                                    risk_reward = trade_params.get('tp_percentage', 0) / trade_params.get('sl_percentage', 1)
                                    
                                    if risk_reward >= 1.5:  # Minimum 1.5:1 risk/reward
                                        print(f"   ✅ Opportunity: {trade_params['direction']} {coin}")
                                        print(f"      Confidence: {trade_params['confidence']}/10 | R:R: 1:{risk_reward:.1f}")
                                        execute_trade(coin, trade_params, market_data[coin]['price'])
                                        opportunities_found += 1
                                        
                                        # Store pattern for learning
                                        pattern_key = f"{coin}_{trade_params['direction']}_{regime_data['regime']}"
                                        pattern_memory[pattern_key] = {
                                            'timeframe_data': multi_tf_data,
                                            'confidence': trade_params['confidence'],
                                            'regime': regime_data['regime'],
                                            'timestamp': current_time
                                        }
                                        
                                        # Limit trades per cycle to manage risk
                                        if opportunities_found >= 2:
                                            print(f"   📊 Trade limit reached (2 per cycle)")
                                            break
                                    else:
                                        print(f"   ⏭️ {coin}: Poor R:R (1:{risk_reward:.1f})")
                                else:
                                    print(f"   ⏭️ {coin}: Low confidence ({trade_params['confidence']}/10)")
                        
                        if opportunities_found == 0:
                            print(f"   No high-confidence opportunities from {opportunities_analyzed} coins analyzed")
                    else:
                        data_ready = min(len(h) for h in price_histories.values())
                        print(f"\n📈 Building price history... ({data_ready}/20 candles)")
                else:
                    print(f"\n⚠️ Position limit reached ({len(portfolio['positions'])}/{MAX_CONCURRENT_POSITIONS})")
                
                # Portfolio performance summary with enhanced metrics
                total_value = calculate_portfolio_value(market_data)
                pnl = total_value - 1000
                pnl_pct = (pnl / 1000) * 100
                
                # Calculate Sharpe ratio (simplified)
                if portfolio['trade_history']:
                    returns = [t['pnl_percent'] for t in portfolio['trade_history'][-20:]]
                    if len(returns) > 1:
                        avg_return = sum(returns) / len(returns)
                        std_dev = math.sqrt(sum((r - avg_return) ** 2 for r in returns) / len(returns))
                        sharpe = (avg_return / std_dev * math.sqrt(252)) if std_dev > 0 else 0
                    else:
                        sharpe = 0
                else:
                    sharpe = 0
                
                print(f"\n💼 PORTFOLIO PERFORMANCE:")
                print(f"   Total Value: ${total_value:.2f}")
                print(f"   P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)")
                print(f"   Free Balance: ${portfolio['balance']:.2f}")
                print(f"   Active Positions: {len(portfolio['positions'])}")
                print(f"   Sharpe Ratio: {sharpe:.2f}")
                
                # Cost tracking with projections
                costs = get_cost_projections()
                print(f"\n💰 COST ANALYSIS:")
                print(f"   Session: ${costs['current']['total']:.4f} ({costs['current']['api_calls']} API calls)")
                print(f"   Weekly Projection: ${costs['projections']['weekly']['total']:.2f}")
                print(f"   Monthly Projection: ${costs['projections']['monthly']['total']:.2f}")
                print(f"   Net Profit (after costs): ${pnl - costs['current']['total']:.2f}")

                if full_analyses_count % 5 == 0:  # every 5 full analyses
                    auto_tune_hyperparameters(window=50)
                
                last_full_analysis = current_time
                print("="*80)
            
            # Sleep for quick check interval
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
    
    # Start enhanced bot
    run_enhanced_bot()