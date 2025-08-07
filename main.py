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
import numpy as np
from collections import deque

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
    
    conn.commit()
    conn.close()
    print("📁 Optimized SQLite database initialized")

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
         confidence, reasoning)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        position.get('reasoning', '')
    ))
    
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
            'reasoning': pos[12]
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
                        'low_24h': float(ticker['lowPrice'])
                    }
        
        return market_data
        
    except Exception as e:
        print(f"Error fetching batch market data: {e}")
        return {}

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
        '5m': ('5m', 24),    # 2 hours of 5-minute candles
        '1h': ('1h', 24),    # 24 hours of hourly candles
        '4h': ('4h', 28)     # ~5 days of 4-hour candles
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
                    'lows': [float(c[3]) for c in candles]
                }
        except:
            pass
    
    return multi_tf_data

def analyze_timeframe_alignment(multi_tf_data):
    """Analyze alignment across timeframes"""
    if not multi_tf_data:
        return {'aligned': False, 'score': 0, 'direction': 'neutral', 'signals': {}}
    
    alignment_score = 0
    signals = {}
    
    for tf, data in multi_tf_data.items():
        if 'closes' in data and len(data['closes']) > 10:
            closes = data['closes']
            
            # Trend detection
            sma_short = sum(closes[-5:]) / 5
            sma_long = sum(closes[-10:]) / 10
            
            if sma_short > sma_long * 1.005:  # 0.5% buffer
                signals[tf] = 'bullish'
                alignment_score += 1
            elif sma_short < sma_long * 0.995:
                signals[tf] = 'bearish'
                alignment_score -= 1
            else:
                signals[tf] = 'neutral'
    
    # Determine overall direction
    if alignment_score >= 2:
        direction = 'bullish'
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
        'signals': signals
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

def smart_position_analysis(position_id, position, market_data, sentiment):
    """Full position analysis with AI (used less frequently)"""
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
    
    # Smart exit analysis only if not hitting SL/TP and significant sentiment shift
    if not should_close and abs(sentiment) > 50:
        # Only use AI if market has shifted significantly
        if (direction == 'LONG' and sentiment < -50) or \
           (direction == 'SHORT' and sentiment > 50):
            
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
    """Optimized AI trading decision"""
    if not client:
        return None
    
    current_price = market_data[coin]['price']
    tf_alignment = analyze_timeframe_alignment(multi_tf_data)
    
    # Skip if timeframes not aligned
    if not tf_alignment['aligned']:
        return {'direction': 'SKIP', 'reason': 'Timeframes not aligned'}
    
    # Simplified prompt for cost efficiency
    analysis_prompt = f"""
{coin} TRADING ANALYSIS

Price: ${current_price:.2f}
24h: {market_data[coin]['change_24h']:+.1f}%
Volume: {market_data[coin]['volume']/1000000:.1f}M

Timeframes: {tf_alignment['direction']} (score: {tf_alignment['score']})
Win Rate: {learning_insights.get('win_rate', 0):.0f}%
Best Leverage: {learning_insights.get('best_leverage', 10)}x

Decision needed:
DECISION: [LONG/SHORT/SKIP]
LEVERAGE: [10-20]
SIZE: [5-10]%
SL: [1.5-3]%
TP: [2-5]%
CONFIDENCE: [1-10]
REASON: [one line]
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": analysis_prompt}],
            max_tokens=150,
            temperature=0.5
        )
        
        # Track cost
        track_cost('openai', 0.00005, '200')
        
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
                    trade_params['leverage'] = min(20, max(10, int(''.join(filter(str.isdigit, value)))))
                elif 'SIZE' in key:
                    size = float(''.join(filter(lambda x: x.isdigit() or x == '.', value)))
                    trade_params['position_size'] = min(0.10, max(0.05, size / 100))
                elif 'SL' in key:
                    sl = float(''.join(filter(lambda x: x.isdigit() or x == '.', value)))
                    trade_params['stop_loss'] = min(0.03, max(0.015, sl / 100))
                elif 'TP' in key:
                    tp = float(''.join(filter(lambda x: x.isdigit() or x == '.', value)))
                    trade_params['take_profit'] = min(0.05, max(0.02, tp / 100))
                elif 'CONFIDENCE' in key:
                    trade_params['confidence'] = min(10, max(1, int(''.join(filter(str.isdigit, value)))))
                elif 'REASON' in key:
                    trade_params['reasoning'] = value[:100]  # Limit reason length
        
        # Default values if parsing fails
        if 'direction' not in trade_params:
            trade_params['direction'] = 'SKIP'
        if 'leverage' not in trade_params:
            trade_params['leverage'] = 10
        if 'position_size' not in trade_params:
            trade_params['position_size'] = 0.05
        if 'stop_loss' not in trade_params:
            trade_params['stop_loss'] = 0.02
        if 'take_profit' not in trade_params:
            trade_params['take_profit'] = 0.03
        if 'confidence' not in trade_params:
            trade_params['confidence'] = 5
        if 'reasoning' not in trade_params:
            trade_params['reasoning'] = 'Standard trade'
        
        trade_params['duration'] = 'SWING'  # Default
        
        return trade_params
        
    except Exception as e:
        print(f"AI Error: {e}")
        return None

def execute_trade(coin, trade_params, current_price):
    """Execute trade with position"""
    global portfolio
    
    if trade_params['direction'] == 'SKIP':
        return
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Position sizing
    position_value = portfolio['balance'] * trade_params['position_size']
    leverage = trade_params['leverage']
    notional_value = position_value * leverage
    
    # Calculate SL/TP prices
    if trade_params['direction'] == 'LONG':
        stop_loss_price = current_price * (1 - trade_params['stop_loss'])
        take_profit_price = current_price * (1 + trade_params['take_profit'])
    else:
        stop_loss_price = current_price * (1 + trade_params['stop_loss'])
        take_profit_price = current_price * (1 - trade_params['take_profit'])
    
    position_id = f"{coin}_{timestamp.replace(' ', '_').replace(':', '-')}"
    
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
        'reasoning': trade_params['reasoning']
    }
    
    portfolio['positions'][position_id] = position
    portfolio['balance'] -= position_value
    
    # Save to database
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
    
    emoji = "📈" if trade_params['direction'] == 'LONG' else "📉"
    print(f"\n   {emoji} {trade_params['direction']} {coin}: ${position_value:.2f} @ {leverage}x")
    print(f"      Confidence: {trade_params['confidence']}/10 | SL: ${stop_loss_price:.2f} | TP: ${take_profit_price:.2f}")

def close_position(position_id, position, current_price, reason, pnl_amount):
    """Close position"""
    global portfolio
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    pnl_percent = (pnl_amount / position['position_size']) * 100
    
    # Return funds
    final_amount = position['position_size'] + pnl_amount
    portfolio['balance'] += max(0, final_amount)
    
    # Save closing trade
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
    
    # Add to history
    portfolio['trade_history'].append({
        'time': timestamp,
        'coin': position['coin'],
        'pnl': pnl_amount,
        'pnl_percent': pnl_percent
    })
    
    # Remove position
    del portfolio['positions'][position_id]
    remove_active_position(position_id)
    
    emoji = "💰" if pnl_amount > 0 else "💔"
    print(f"   {emoji} CLOSED {position['coin']}: ${pnl_amount:+.2f} ({pnl_percent:+.1f}%) - {reason}")

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
    
    conn.close()
    return insights

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

def run_optimized_bot():
    """Main bot loop with TRUE hybrid monitoring approach"""
    # Initialize
    init_database()
    portfolio['positions'] = load_active_positions()
    
    print("🚀 OPTIMIZED SMART TRADING BOT - HYBRID MODE")
    print("⚡ Quick checks: Every 15 seconds (positions only)")
    print("🧠 Full analysis: Every 2 minutes (complete scan)")
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
    iteration = 0
    quick_checks_count = 0
    full_analyses_count = 0
    
    while True:
        try:
            iteration += 1
            current_time = time.time()
            
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
                print(f"Quick checks since last analysis: {quick_checks_count}")
                print("="*80)
                
                quick_checks_count = 0  # Reset counter
                
                # Get comprehensive market data (batch call)
                market_data = get_batch_market_data()
                
                # Update price histories for all coins
                for coin in target_coins:
                    if coin in market_data:
                        price_histories[coin].append(market_data[coin]['price'])
                        if len(price_histories[coin]) > 50:
                            price_histories[coin] = price_histories[coin][-50:]
                
                # Display market overview with technical indicators
                print("\n📊 Market Overview:")
                market_summary = []
                for coin in target_coins[:6]:  # Show first 6 coins
                    if coin in market_data:
                        data = market_data[coin]
                        rsi = calculate_rsi(price_histories[coin]) if len(price_histories[coin]) > 14 else 50
                        
                        # Determine trend
                        trend = "→"
                        if data['change_24h'] > 1:
                            trend = "↑"
                        elif data['change_24h'] < -1:
                            trend = "↓"
                        
                        print(f"   {coin}: ${data['price']:.2f} {trend} ({data['change_24h']:+.1f}%) RSI:{rsi:.0f}")
                
                # Advanced position management with sentiment analysis
                if portfolio['positions']:
                    print(f"\n📈 Position Management:")
                    sentiment_scores = {}
                    
                    # Calculate sentiment for all coins
                    for coin in market_data:
                        rsi = calculate_rsi(price_histories[coin]) if len(price_histories[coin]) > 14 else 50
                        momentum = market_data[coin]['change_24h']
                        
                        # Complex sentiment calculation
                        sentiment = 0
                        if rsi < 30:
                            sentiment += 40  # Oversold
                        elif rsi > 70:
                            sentiment -= 40  # Overbought
                        else:
                            sentiment += (50 - rsi) * 0.5
                        
                        sentiment += min(max(momentum * 2, -30), 30)
                        sentiment_scores[coin] = sentiment
                    
                    # Analyze each position with full context
                    for pos_id, position in list(portfolio['positions'].items()):
                        coin = position['coin']
                        if coin in market_data:
                            current_price = market_data[coin]['price']
                            entry_price = position['entry_price']
                            
                            # Calculate current P&L
                            if position['direction'] == 'LONG':
                                pnl_pct = (current_price - entry_price) / entry_price
                            else:
                                pnl_pct = (entry_price - current_price) / entry_price
                            
                            pnl_amount = pnl_pct * position['notional_value']
                            
                            print(f"   • {position['direction']} {coin}: P&L ${pnl_amount:+.2f} ({pnl_pct*100:+.1f}%) | Sentiment: {sentiment_scores.get(coin, 0):.0f}")
                            
                            # Smart position analysis with AI
                            smart_position_analysis(pos_id, position, market_data, sentiment_scores.get(coin, 0))
                
                # Look for new trading opportunities
                if len(portfolio['positions']) < MAX_CONCURRENT_POSITIONS:
                    if all(len(history) >= 20 for history in price_histories.values()):
                        learning_insights = get_learning_insights()
                        
                        print(f"\n🤖 AI Opportunity Scan:")
                        print(f"   Historical Win Rate: {learning_insights['win_rate']:.1f}%")
                        print(f"   Optimal Leverage: {learning_insights.get('best_leverage', 10)}x")
                        print(f"   Scanning {len(target_coins)} coins...")
                        
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
                            
                            # AI trade decision with all context
                            trade_params = ai_trade_decision(coin, market_data, multi_tf_data, learning_insights)
                            
                            if trade_params and trade_params['direction'] != 'SKIP':
                                if trade_params['confidence'] >= MIN_CONFIDENCE_THRESHOLD:
                                    print(f"   ✅ Opportunity: {trade_params['direction']} {coin} (Confidence: {trade_params['confidence']}/10)")
                                    execute_trade(coin, trade_params, market_data[coin]['price'])
                                    opportunities_found += 1
                                    
                                    # Store pattern for learning
                                    pattern_key = f"{coin}_{trade_params['direction']}"
                                    pattern_memory[pattern_key] = {
                                        'timeframe_data': multi_tf_data,
                                        'confidence': trade_params['confidence'],
                                        'timestamp': current_time
                                    }
                                    
                                    # Limit trades per cycle to manage risk
                                    if opportunities_found >= 2:
                                        print(f"   📊 Trade limit reached (2 per cycle)")
                                        break
                                else:
                                    print(f"   ⏭️ {coin}: Low confidence ({trade_params['confidence']}/10)")
                        
                        if opportunities_found == 0:
                            print(f"   No high-confidence opportunities from {opportunities_analyzed} coins analyzed")
                    else:
                        data_ready = min(len(h) for h in price_histories.values())
                        print(f"\n📈 Building price history... ({data_ready}/20 candles)")
                else:
                    print(f"\n⚠️ Position limit reached ({len(portfolio['positions'])}/{MAX_CONCURRENT_POSITIONS})")
                
                # Portfolio performance summary
                total_value = calculate_portfolio_value(market_data)
                pnl = total_value - 1000
                pnl_pct = (pnl / 1000) * 100
                
                print(f"\n💼 PORTFOLIO PERFORMANCE:")
                print(f"   Total Value: ${total_value:.2f}")
                print(f"   P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)")
                print(f"   Free Balance: ${portfolio['balance']:.2f}")
                print(f"   Active Positions: {len(portfolio['positions'])}")
                
                # Cost tracking with projections
                costs = get_cost_projections()
                print(f"\n💰 COST ANALYSIS:")
                print(f"   Session: ${costs['current']['total']:.4f} ({costs['current']['api_calls']} API calls)")
                print(f"   Weekly Projection: ${costs['projections']['weekly']['total']:.2f}")
                print(f"   Monthly Projection: ${costs['projections']['monthly']['total']:.2f}")
                print(f"   Net Profit (after costs): ${pnl - costs['current']['total']:.2f}")
                
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
    
    # Start optimized bot
    run_optimized_bot()