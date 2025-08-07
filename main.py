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

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

# Database setup
def init_database():
    """Initialize SQLite database with trading tables"""
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    
    # Trades table
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
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
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
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Performance metrics table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_name TEXT UNIQUE NOT NULL,
            metric_value REAL NOT NULL,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("📁 SQLite database initialized")

def save_trade_to_db(trade_record):
    """Save trade record to database"""
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

def save_learning_pattern_to_db(pattern):
    """Save learning pattern to database"""
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO learning_patterns 
        (coin, direction, leverage, confidence, duration_target, profitable, 
         pnl_percent, duration_actual, market_conditions)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        pattern['coin'],
        pattern['direction'],
        pattern['leverage'],
        pattern['entry_conditions']['confidence'],
        pattern['entry_conditions']['duration_target'],
        pattern['outcome']['profitable'],
        pattern['outcome']['pnl_percent'],
        str(pattern['outcome']['duration']),
        json.dumps(pattern.get('market_conditions', {}))
    ))
    
    conn.commit()
    conn.close()

def get_learning_insights():
    """Get AI learning insights from database"""
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    
    insights = {}
    
    # Win rate
    cursor.execute('SELECT COUNT(*) FROM trades WHERE pnl IS NOT NULL')
    total_trades = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM trades WHERE pnl > 0')
    winning_trades = cursor.fetchone()[0]
    
    insights['win_rate'] = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    insights['total_trades'] = total_trades
    
    # Best performing leverage
    cursor.execute('''
        SELECT leverage, AVG(pnl) as avg_pnl, COUNT(*) as count 
        FROM trades 
        WHERE pnl IS NOT NULL AND leverage IS NOT NULL
        GROUP BY leverage 
        ORDER BY avg_pnl DESC
    ''')
    leverage_performance = cursor.fetchall()
    insights['best_leverage'] = leverage_performance[0][0] if leverage_performance else 10
    
    # Best performing coins
    cursor.execute('''
        SELECT coin, AVG(pnl) as avg_pnl, COUNT(*) as count,
               SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate
        FROM trades 
        WHERE pnl IS NOT NULL 
        GROUP BY coin 
        ORDER BY avg_pnl DESC
    ''')
    coin_performance = cursor.fetchall()
    insights['coin_performance'] = coin_performance
    
    # Average profit/loss
    cursor.execute('SELECT AVG(pnl) FROM trades WHERE pnl > 0')
    avg_profit = cursor.fetchone()[0]
    insights['avg_profit'] = avg_profit if avg_profit else 0
    
    cursor.execute('SELECT AVG(pnl) FROM trades WHERE pnl < 0')
    avg_loss = cursor.fetchone()[0]
    insights['avg_loss'] = avg_loss if avg_loss else 0
    
    # Recent performance trends
    cursor.execute('''
        SELECT DATE(created_at) as trade_date, 
               SUM(pnl) as daily_pnl,
               COUNT(*) as daily_trades
        FROM trades 
        WHERE pnl IS NOT NULL AND created_at >= datetime('now', '-7 days')
        GROUP BY DATE(created_at)
        ORDER BY trade_date DESC
    ''')
    recent_performance = cursor.fetchall()
    insights['recent_performance'] = recent_performance
    
    conn.close()
    return insights

def load_portfolio_from_db():
    """Load existing portfolio state from database"""
    global portfolio
    
    # Load recent trade history (last 50 trades)
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM trades 
        ORDER BY created_at DESC 
        LIMIT 50
    ''')
    
    db_trades = cursor.fetchall()
    columns = [description[0] for description in cursor.description]
    
    portfolio['trade_history'] = []
    for trade in db_trades:
        trade_dict = dict(zip(columns, trade))
        portfolio['trade_history'].append({
            'time': trade_dict['timestamp'],
            'position_id': trade_dict['position_id'],
            'coin': trade_dict['coin'],
            'action': trade_dict['action'],
            'price': trade_dict['price'],
            'position_size': trade_dict['position_size'],
            'pnl': trade_dict['pnl'],
            'pnl_percent': trade_dict['pnl_percent'],
            'reason': trade_dict['reason']
        })
    
    conn.close()
    print(f"📊 Loaded {len(portfolio['trade_history'])} previous trades from database")

# Advanced Leverage Trading Portfolio
portfolio = {
    'balance': 1000.0,  # Starting with $1000 virtual money
    'positions': {},    # More complex position structure for leverage
    'trade_history': [],
    'learning_data': {
        'successful_patterns': [],
        'failed_patterns': [],
        'performance_metrics': {
            'win_rate': 0,
            'avg_profit': 0,
            'avg_loss': 0,
            'best_leverage': 10,
            'best_timeframes': []
        }
    }
}

# Flask web app for dashboard
app = Flask(__name__)

@app.route('/')
def dashboard():
    """Serve the main dashboard"""
    return send_from_directory('.', 'index.html')

@app.route('/dashboard.js')
def dashboard_js():
    """Serve the JavaScript file"""
    return send_from_directory('.', 'dashboard.js')

@app.route('/api/status')
def api_status():
    """API endpoint for dashboard data"""
    try:
        current_market_data = get_market_data()
        total_value = calculate_portfolio_value(current_market_data)
        
        # Get learning insights from database
        learning_insights = get_learning_insights()
        
        response_data = {
            'total_value': total_value,
            'balance': portfolio['balance'],
            'positions': portfolio['positions'],
            'trade_history': portfolio['trade_history'][-20:],  # Last 20 trades
            'market_data': current_market_data,
            'learning_metrics': learning_insights,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics')
def api_analytics():
    """Advanced analytics endpoint"""
    try:
        conn = sqlite3.connect('trading_bot.db')
        cursor = conn.cursor()
        
        # Performance over time
        cursor.execute('''
            SELECT DATE(created_at) as date, 
                   SUM(pnl) as daily_pnl,
                   COUNT(*) as trades_count,
                   AVG(pnl) as avg_pnl
            FROM trades 
            WHERE pnl IS NOT NULL 
            GROUP BY DATE(created_at) 
            ORDER BY date DESC 
            LIMIT 30
        ''')
        performance_data = cursor.fetchall()
        
        # Leverage analysis
        cursor.execute('''
            SELECT leverage, 
                   COUNT(*) as trade_count,
                   AVG(pnl) as avg_pnl,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate
            FROM trades 
            WHERE pnl IS NOT NULL AND leverage IS NOT NULL
            GROUP BY leverage
        ''')
        leverage_analysis = cursor.fetchall()
        
        conn.close()
        
        return jsonify({
            'performance_over_time': performance_data,
            'leverage_analysis': leverage_analysis
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_flask_app():
    """Run Flask in a separate thread"""
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)

def get_market_data():
    """Get comprehensive market data"""
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']
    market_data = {}
    
    for symbol in symbols:
        coin = symbol.replace('USDT', '')
        
        # 24hr ticker data
        ticker_url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
        ticker = requests.get(ticker_url).json()
        
        # Order book depth
        depth_url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=20"
        depth = requests.get(depth_url).json()
        
        # Recent trades
        trades_url = f"https://api.binance.com/api/v3/trades?symbol={symbol}&limit=100"
        trades = requests.get(trades_url).json()
        
        market_data[coin] = {
            'price': float(ticker['lastPrice']),
            'change_24h': float(ticker['priceChangePercent']),
            'volume': float(ticker['volume']),
            'high_24h': float(ticker['highPrice']),
            'low_24h': float(ticker['lowPrice']),
            'order_book': depth,
            'recent_trades': trades
        }
    
    return market_data

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

def analyze_market_sentiment(market_data, price_histories):
    """Advanced market sentiment analysis"""
    sentiment_scores = {}
    
    for coin, data in market_data.items():
        rsi = calculate_rsi(price_histories.get(coin, []))
        price_momentum = data['change_24h']
        volume_ratio = data['volume'] / statistics.mean([d['volume'] for d in market_data.values()])
        
        # Calculate comprehensive sentiment score (-100 to +100)
        sentiment = 0
        
        # RSI contribution
        if rsi < 30:
            sentiment += 40  # Oversold, bullish
        elif rsi > 70:
            sentiment -= 40  # Overbought, bearish
        else:
            sentiment += (50 - rsi) * 0.5  # Neutral zone
        
        # Price momentum contribution
        sentiment += min(max(price_momentum * 2, -30), 30)
        
        # Volume contribution
        if volume_ratio > 1.5:
            sentiment += 15  # High volume confirms trend
        elif volume_ratio < 0.5:
            sentiment -= 10  # Low volume weakens signal
        
        sentiment_scores[coin] = max(-100, min(100, sentiment))
    
    return sentiment_scores

def ai_determine_trade_params(coin, market_data, sentiment, price_histories, learning_insights):
    """AI determines optimal trade parameters using database insights"""
    if not client:
        return None
    
    current_price = market_data[coin]['price']
    rsi = calculate_rsi(price_histories.get(coin, []))
    
    # Get recent price action for context
    recent_prices = price_histories.get(coin, [])[-20:] if coin in price_histories else []
    price_volatility = statistics.stdev(recent_prices) if len(recent_prices) > 5 else 0
    
    # Use learning insights for better decisions
    coin_performance = None
    for perf in learning_insights.get('coin_performance', []):
        if perf[0] == coin:
            coin_performance = perf
            break
    
    # Build comprehensive analysis with learning data
    analysis_prompt = f"""
ADVANCED LEVERAGE TRADING ANALYSIS for {coin}

CURRENT MARKET DATA:
Price: ${current_price:,.2f}
24h Change: {market_data[coin]['change_24h']:+.2f}%
RSI: {rsi:.1f}
Sentiment Score: {sentiment:.1f}/100
Price Volatility: {price_volatility:.2f}
Volume: {market_data[coin]['volume']:,.0f}

AI LEARNING INSIGHTS:
Overall Win Rate: {learning_insights.get('win_rate', 0):.1f}%
Total Historical Trades: {learning_insights.get('total_trades', 0)}
Best Performing Leverage: {learning_insights.get('best_leverage', 10)}x
Average Profit: ${learning_insights.get('avg_profit', 0):.2f}
Average Loss: ${learning_insights.get('avg_loss', 0):.2f}

{coin} HISTORICAL PERFORMANCE:
{f"Average P&L: ${coin_performance[1]:.2f}" if coin_performance else "No historical data"}
{f"Win Rate: {coin_performance[3]:.1f}%" if coin_performance else ""}
{f"Trade Count: {coin_performance[2]}" if coin_performance else ""}

RECENT PERFORMANCE TREND:
{learning_insights.get('recent_performance', [])[:3]}

ENHANCED ANALYSIS REQUIREMENTS:
1. Trade Direction: LONG (bullish) or SHORT (bearish) or SKIP
2. Leverage: 10-20x (use learning insights for optimization)
3. Position Size: 5-10% (adjust based on historical performance)
4. Stop Loss: Dynamic based on volatility and learned patterns
5. Take Profit: Optimize based on historical success rates
6. Trade Duration: SCALP/SWING/POSITION based on what worked before

LEARNING-BASED DECISION FACTORS:
- If this coin historically underperforms, reduce position size
- If recent trades are losing, be more conservative
- Use best performing leverage from database
- Apply lessons from similar market conditions

Provide analysis in this format:
DECISION: [LONG/SHORT/SKIP]
LEVERAGE: [10-20]x
POSITION_SIZE: [5-10]%
STOP_LOSS: [percentage from entry]
TAKE_PROFIT: [percentage from entry]
DURATION: [SCALP/SWING/POSITION]
CONFIDENCE: [1-10]
REASONING: [Brief explanation with learning insights]
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": analysis_prompt}],
            max_tokens=250
        )
        
        ai_response = response.choices[0].message.content.strip()
        
        # Parse AI response
        trade_params = {}
        for line in ai_response.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                if key == 'decision':
                    trade_params['direction'] = value
                elif key == 'leverage':
                    trade_params['leverage'] = int(value.replace('x', ''))
                elif key == 'position_size':
                    trade_params['position_size'] = float(value.replace('%', '')) / 100
                elif key == 'stop_loss':
                    trade_params['stop_loss'] = float(value.replace('%', '')) / 100
                elif key == 'take_profit':
                    trade_params['take_profit'] = float(value.replace('%', '')) / 100
                elif key == 'duration':
                    trade_params['duration'] = value
                elif key == 'confidence':
                    trade_params['confidence'] = int(value)
                elif key == 'reasoning':
                    trade_params['reasoning'] = value
        
        return trade_params if 'direction' in trade_params else None
        
    except Exception as e:
        print(f"AI Analysis Error: {e}")
        return None

def execute_leverage_trade(coin, trade_params, current_price):
    """Execute leverage trade with AI parameters"""
    global portfolio
    
    if trade_params['direction'] == 'SKIP':
        return
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Calculate position details
    position_value = portfolio['balance'] * trade_params['position_size']
    leverage = trade_params['leverage']
    notional_value = position_value * leverage
    
    # Calculate stop loss and take profit prices
    if trade_params['direction'] == 'LONG':
        stop_loss_price = current_price * (1 - trade_params['stop_loss'])
        take_profit_price = current_price * (1 + trade_params['take_profit'])
    else:  # SHORT
        stop_loss_price = current_price * (1 + trade_params['stop_loss'])
        take_profit_price = current_price * (1 - trade_params['take_profit'])
    
    # Create position
    position_id = f"{coin}_{timestamp.replace(' ', '_').replace(':', '-')}"
    
    portfolio['positions'][position_id] = {
        'coin': coin,
        'direction': trade_params['direction'],
        'entry_price': current_price,
        'entry_time': timestamp,
        'position_size': position_value,
        'leverage': leverage,
        'notional_value': notional_value,
        'stop_loss': stop_loss_price,
        'take_profit': take_profit_price,
        'duration_target': trade_params['duration'],
        'confidence': trade_params['confidence'],
        'reasoning': trade_params['reasoning']
    }
    
    # Reserve margin (simplified - in reality, exchanges calculate this differently)
    portfolio['balance'] -= position_value
    
    # Record trade
    trade_record = {
        'time': timestamp,
        'position_id': position_id,
        'coin': coin,
        'action': f"{trade_params['direction']} OPEN",
        'price': current_price,
        'position_size': position_value,
        'leverage': leverage,
        'notional_value': notional_value,
        'stop_loss': stop_loss_price,
        'take_profit': take_profit_price,
        'reasoning': trade_params['reasoning']
    }
    # Record trade in database
    save_trade_to_db(trade_record)
    
    direction_emoji = "📈" if trade_params['direction'] == 'LONG' else "📉"
    print(f"   {direction_emoji} OPENED {trade_params['direction']} {coin}: ${position_value:.2f} @ {leverage}x leverage")
    print(f"      Entry: ${current_price:,.2f} | Notional: ${notional_value:,.2f}")
    print(f"      Stop Loss: ${stop_loss_price:,.2f} | Take Profit: ${take_profit_price:,.2f}")
    print(f"      Duration: {trade_params['duration']} | Confidence: {trade_params['confidence']}/10")

def check_position_management(position_id, position, current_price, market_sentiment):
    """Advanced position management with AI monitoring"""
    global portfolio
    
    coin = position['coin']
    entry_price = position['entry_price']
    direction = position['direction']
    
    # Calculate current P&L
    if direction == 'LONG':
        pnl_percent = (current_price - entry_price) / entry_price
    else:  # SHORT
        pnl_percent = (entry_price - current_price) / entry_price
    
    pnl_amount = pnl_percent * position['notional_value']
    
    # Check stop loss and take profit
    should_close = False
    close_reason = ""
    
    if direction == 'LONG':
        if current_price <= position['stop_loss']:
            should_close = True
            close_reason = "Stop Loss Hit"
        elif current_price >= position['take_profit']:
            should_close = True
            close_reason = "Take Profit Hit"
    else:  # SHORT
        if current_price >= position['stop_loss']:
            should_close = True
            close_reason = "Stop Loss Hit"
        elif current_price <= position['take_profit']:
            should_close = True
            close_reason = "Take Profit Hit"
    
    # AI-based early exit if market conditions change
    if not should_close and market_sentiment is not None:
        sentiment_threshold = 30 if direction == 'LONG' else -30
        
        if ((direction == 'LONG' and market_sentiment < -sentiment_threshold) or
            (direction == 'SHORT' and market_sentiment > sentiment_threshold)):
            
            # Additional AI check for early exit
            if client:
                try:
                    exit_prompt = f"""
POSITION MANAGEMENT ANALYSIS

Current Position: {direction} {coin} at ${entry_price:,.2f}
Current Price: ${current_price:,.2f}
Current P&L: ${pnl_amount:+.2f} ({pnl_percent*100:+.2f}%)
Market Sentiment: {market_sentiment:.1f}/100
Position Age: {datetime.now() - datetime.strptime(position['entry_time'], '%Y-%m-%d %H:%M:%S')}

DECISION NEEDED: Should we close this position early to prevent larger losses?

Consider:
- Current unrealized P&L
- Market sentiment shift
- Risk of further losses
- Better opportunities elsewhere

Answer: CLOSE (with reason) or HOLD (with reason)
"""
                    
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": exit_prompt}],
                        max_tokens=100
                    )
                    
                    ai_decision = response.choices[0].message.content.strip()
                    
                    if "CLOSE" in ai_decision.upper():
                        should_close = True
                        close_reason = f"AI Early Exit: {ai_decision}"
                        
                except Exception as e:
                    print(f"AI Exit Analysis Error: {e}")
    
    # Close position if needed
    if should_close:
        close_position(position_id, position, current_price, close_reason, pnl_amount)
    
    return pnl_amount

def close_position(position_id, position, current_price, reason, pnl_amount):
    """Close leverage position"""
    global portfolio
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Return margin plus/minus P&L
    final_amount = position['position_size'] + pnl_amount
    portfolio['balance'] += final_amount
    
    # Record closing trade
    trade_record = {
        'time': timestamp,
        'position_id': position_id,
        'coin': position['coin'],
        'action': f"{position['direction']} CLOSE",
        'price': current_price,
        'position_size': position['position_size'],
        'pnl': pnl_amount,
        'pnl_percent': (pnl_amount / position['position_size']) * 100,
        'reason': reason,
        'duration': str(datetime.now() - datetime.strptime(position['entry_time'], '%Y-%m-%d %H:%M:%S'))
    }
    # Record closing trade in database
    save_trade_to_db(trade_record)
    
    # Add to in-memory history for immediate dashboard access
    portfolio['trade_history'].append(trade_record)
    
    # Learn from the trade
    learn_from_trade(position, trade_record)
    
    # Remove position
    del portfolio['positions'][position_id]
    
    pnl_emoji = "💰" if pnl_amount > 0 else "📉"
    print(f"   {pnl_emoji} CLOSED {position['direction']} {position['coin']}: P&L ${pnl_amount:+.2f}")
    print(f"      Reason: {reason}")

def learn_from_trade(position, trade_record):
    """Machine learning from completed trades with database storage"""
    was_profitable = trade_record['pnl'] > 0
    
    # Extract trade pattern
    trade_pattern = {
        'coin': position['coin'],
        'direction': position['direction'],
        'leverage': position['leverage'],
        'entry_conditions': {
            'confidence': position['confidence'],
            'duration_target': position['duration_target']
        },
        'outcome': {
            'profitable': was_profitable,
            'pnl_percent': trade_record['pnl_percent'],
            'duration': trade_record['duration']
        }
    }
    
    # Save pattern to database
    save_learning_pattern_to_db(trade_pattern)
    
    # Update in-memory learning data for immediate use
    if was_profitable:
        portfolio['learning_data']['successful_patterns'].append(trade_pattern)
        if len(portfolio['learning_data']['successful_patterns']) > 50:
            portfolio['learning_data']['successful_patterns'] = portfolio['learning_data']['successful_patterns'][-50:]
    else:
        portfolio['learning_data']['failed_patterns'].append(trade_pattern)
        if len(portfolio['learning_data']['failed_patterns']) > 50:
            portfolio['learning_data']['failed_patterns'] = portfolio['learning_data']['failed_patterns'][-50:]

def update_performance_metrics():
    """Update learning performance metrics from database"""
    learning_insights = get_learning_insights()
    portfolio['learning_data']['performance_metrics'] = {
        'win_rate': learning_insights.get('win_rate', 0),
        'avg_profit': learning_insights.get('avg_profit', 0),
        'avg_loss': learning_insights.get('avg_loss', 0),
        'best_leverage': learning_insights.get('best_leverage', 10),
        'total_trades': learning_insights.get('total_trades', 0)
    }

def calculate_portfolio_value(market_data):
    """Calculate total portfolio value including leveraged positions"""
    total_value = portfolio['balance']
    
    for position_id, position in portfolio['positions'].items():
        coin = position['coin']
        if coin in market_data:
            current_price = market_data[coin]['price']
            entry_price = position['entry_price']
            direction = position['direction']
            
            # Calculate P&L
            if direction == 'LONG':
                pnl_percent = (current_price - entry_price) / entry_price
            else:  # SHORT
                pnl_percent = (entry_price - current_price) / entry_price
            
            pnl_amount = pnl_percent * position['notional_value']
            position_value = position['position_size'] + pnl_amount
            total_value += position_value
    
    return total_value

def display_portfolio_status(market_data):
    """Display current portfolio status with leverage details"""
    total_value = calculate_portfolio_value(market_data)
    profit_loss = total_value - 1000
    
    print(f"\n💼 LEVERAGE PORTFOLIO STATUS:")
    print(f"   Cash Balance: ${portfolio['balance']:.2f}")
    print(f"   Total Value: ${total_value:.2f}")
    print(f"   P&L: ${profit_loss:+.2f} ({(profit_loss/1000)*100:+.2f}%)")
    print(f"   Active Positions: {len(portfolio['positions'])}")
    print(f"   Total Trades: {len(portfolio['trade_history'])}")
    
    # Learning metrics
    metrics = portfolio['learning_data']['performance_metrics']
    print(f"   Win Rate: {metrics['win_rate']:.1f}% | Best Leverage: {metrics['best_leverage']}x")
    
    if portfolio['positions']:
        print(f"   📊 Open Leveraged Positions:")
        for pos_id, position in portfolio['positions'].items():
            coin = position['coin']
            current_price = market_data[coin]['price']
            entry_price = position['entry_price']
            direction = position['direction']
            leverage = position['leverage']
            
            if direction == 'LONG':
                pnl_percent = (current_price - entry_price) / entry_price
            else:
                pnl_percent = (entry_price - current_price) / entry_price
            
            pnl_amount = pnl_percent * position['notional_value']
            
            print(f"      {direction} {coin} @ {leverage}x: ${position['position_size']:.2f}")
            print(f"           Entry: ${entry_price:,.2f} | Current: ${current_price:,.2f}")
            print(f"           Unrealized P&L: ${pnl_amount:+.2f} ({pnl_percent*100:+.2f}%)")

def comprehensive_market_analysis(market_data, price_histories):
    """Comprehensive AI analysis for all coins using database insights"""
    sentiment_scores = analyze_market_sentiment(market_data, price_histories)
    learning_insights = get_learning_insights()  # Get insights from database
    
    trading_opportunities = {}
    
    for coin in market_data.keys():
        # Skip if already have position in this coin
        coin_positions = [p for p in portfolio['positions'].values() if p['coin'] == coin]
        if len(coin_positions) >= 1:  # Max 1 position per coin
            continue
            
        trade_params = ai_determine_trade_params(
            coin, market_data, sentiment_scores[coin], price_histories, learning_insights
        )
        
        if trade_params and trade_params['direction'] != 'SKIP':
            trading_opportunities[coin] = trade_params
    
    return trading_opportunities, sentiment_scores

def run_trading_bot():
    """Advanced leverage trading bot loop with database persistence"""
    # Initialize database
    init_database()
    
    # Load existing data from database
    load_portfolio_from_db()
    
    price_histories = {'BTC': [], 'ETH': [], 'SOL': [], 'BNB': []}
    
    print("🚀 AI LEVERAGE TRADING BOT STARTED")
    print("💰 Starting Balance: $1,000")
    print("⚡ Leverage: 10-20x based on AI confidence")
    print("🧠 AI Learning: Adapting from every trade")
    print("📁 SQLite Database: Persistent learning enabled")
    print("🌐 Dashboard available at: http://localhost:5000")
    
    while True:
        try:
            print("\n" + "="*80)
            print(f"📊 LEVERAGE TRADING SCAN - {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)
            
            market_data = get_market_data()
            
            # Update price histories
            for coin in market_data:
                price_histories[coin].append(market_data[coin]['price'])
                if len(price_histories[coin]) > 50:
                    price_histories[coin] = price_histories[coin][-50:]
            
            # Display market overview
            for coin, data in market_data.items():
                rsi = calculate_rsi(price_histories[coin])
                print(f"💰 {coin}: ${data['price']:,.2f} ({data['change_24h']:+.2f}%) | RSI: {rsi:.1f}")
            
            # Manage existing positions
            sentiment_scores = analyze_market_sentiment(market_data, price_histories)
            
            for pos_id, position in list(portfolio['positions'].items()):
                coin = position['coin']
                current_price = market_data[coin]['price']
                sentiment = sentiment_scores.get(coin, 0)
                check_position_management(pos_id, position, current_price, sentiment)
            
            # Look for new trading opportunities
            if all(len(history) >= 10 for history in price_histories.values()):
                opportunities, sentiment_scores = comprehensive_market_analysis(market_data, price_histories)
                
                if opportunities:
                    print(f"\n🤖 AI LEVERAGE ANALYSIS:")
                    for coin, params in opportunities.items():
                        direction_emoji = "🟢" if params['direction'] == 'LONG' else "🔴"
                        print(f"   {direction_emoji} {coin}: {params['direction']} @ {params['leverage']}x")
                        print(f"      Confidence: {params['confidence']}/10 | Size: {params['position_size']*100:.1f}%")
                        
                        # Execute the trade
                        current_price = market_data[coin]['price']
                        execute_leverage_trade(coin, params, current_price)
                else:
                    print(f"\n🤖 AI ANALYSIS: No high-confidence opportunities found")
                    
                # Update performance metrics from database
                update_performance_metrics()
            else:
                data_count = min(len(h) for h in price_histories.values())
                print(f"\n📈 Building AI database... ({data_count}/10 data points)")
            
            display_portfolio_status(market_data)
            print("="*80)
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        time.sleep(120)  # Analysis every 2 minutes

# Start both Flask web server and trading bot
if __name__ == "__main__":
    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=run_flask_app, daemon=True)
    flask_thread.start()
    
    # Start the advanced leverage trading bot
    run_trading_bot()