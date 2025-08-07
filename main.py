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

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

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
        
        response_data = {
            'total_value': total_value,
            'balance': portfolio['balance'],
            'positions': portfolio['positions'],
            'trade_history': portfolio['trade_history'],
            'market_data': current_market_data,
            'learning_metrics': portfolio['learning_data']['performance_metrics'],
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response_data)
    
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

def ai_determine_trade_params(coin, market_data, sentiment, price_histories, learning_data):
    """AI determines optimal trade parameters"""
    if not client:
        return None
    
    current_price = market_data[coin]['price']
    rsi = calculate_rsi(price_histories.get(coin, []))
    
    # Get recent price action for context
    recent_prices = price_histories.get(coin, [])[-20:] if coin in price_histories else []
    price_volatility = statistics.stdev(recent_prices) if len(recent_prices) > 5 else 0
    
    # Build comprehensive analysis
    analysis_prompt = f"""
LEVERAGE TRADING ANALYSIS for {coin}

MARKET DATA:
Price: ${current_price:,.2f}
24h Change: {market_data[coin]['change_24h']:+.2f}%
RSI: {rsi:.1f}
Sentiment Score: {sentiment:.1f}/100
Price Volatility: {price_volatility:.2f}
Volume: {market_data[coin]['volume']:,.0f}

LEARNING DATA:
Historical Win Rate: {learning_data['performance_metrics']['win_rate']:.1f}%
Best Performing Leverage: {learning_data['performance_metrics']['best_leverage']}x
Average Profit: ${learning_data['performance_metrics']['avg_profit']:.2f}
Average Loss: ${learning_data['performance_metrics']['avg_loss']:.2f}

ANALYSIS REQUIREMENTS:
1. Trade Direction: LONG (bullish) or SHORT (bearish) or SKIP
2. Leverage: 10-20x based on opportunity strength
3. Position Size: 5-10% of portfolio based on confidence
4. Stop Loss: Dynamic based on volatility and risk
5. Take Profit: Risk/reward ratio of at least 2:1
6. Trade Duration: SCALP (minutes), SWING (hours), POSITION (days)

Consider:
- Strong bullish signals: RSI <30, positive momentum, high volume
- Strong bearish signals: RSI >70, negative momentum, high volume
- High volatility = lower leverage, tighter stops
- Use learning data to avoid repeated mistakes

Provide analysis in this format:
DECISION: [LONG/SHORT/SKIP]
LEVERAGE: [10-20]x
POSITION_SIZE: [5-10]%
STOP_LOSS: [percentage from entry]
TAKE_PROFIT: [percentage from entry]
DURATION: [SCALP/SWING/POSITION]
CONFIDENCE: [1-10]
REASONING: [Brief explanation]
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": analysis_prompt}],
            max_tokens=200
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
    portfolio['trade_history'].append(trade_record)
    
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
    portfolio['trade_history'].append(trade_record)
    
    # Learn from the trade
    learn_from_trade(position, trade_record)
    
    # Remove position
    del portfolio['positions'][position_id]
    
    pnl_emoji = "💰" if pnl_amount > 0 else "📉"
    print(f"   {pnl_emoji} CLOSED {position['direction']} {position['coin']}: P&L ${pnl_amount:+.2f}")
    print(f"      Reason: {reason}")

def learn_from_trade(position, trade_record):
    """Machine learning from completed trades"""
    global portfolio
    
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
    
    # Store pattern
    if was_profitable:
        portfolio['learning_data']['successful_patterns'].append(trade_pattern)
    else:
        portfolio['learning_data']['failed_patterns'].append(trade_pattern)
    
    # Update performance metrics
    update_performance_metrics()

def update_performance_metrics():
    """Update learning performance metrics"""
    global portfolio
    
    all_trades = [t for t in portfolio['trade_history'] if 'pnl' in t]
    
    if len(all_trades) == 0:
        return
    
    profitable_trades = [t for t in all_trades if t['pnl'] > 0]
    losing_trades = [t for t in all_trades if t['pnl'] < 0]
    
    metrics = portfolio['learning_data']['performance_metrics']
    
    # Win rate
    metrics['win_rate'] = (len(profitable_trades) / len(all_trades)) * 100
    
    # Average profit/loss
    if profitable_trades:
        metrics['avg_profit'] = statistics.mean([t['pnl'] for t in profitable_trades])
    if losing_trades:
        metrics['avg_loss'] = statistics.mean([t['pnl'] for t in losing_trades])
    
    # Best leverage (most profitable on average)
    leverage_performance = {}
    for trade in all_trades:
        lev = trade.get('leverage', 10)
        if lev not in leverage_performance:
            leverage_performance[lev] = []
        leverage_performance[lev].append(trade['pnl'])
    
    if leverage_performance:
        best_lev = max(leverage_performance.keys(), 
                      key=lambda x: statistics.mean(leverage_performance[x]))
        metrics['best_leverage'] = best_lev

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
    """Comprehensive AI analysis for all coins"""
    sentiment_scores = analyze_market_sentiment(market_data, price_histories)
    learning_data = portfolio['learning_data']
    
    trading_opportunities = {}
    
    for coin in market_data.keys():
        # Skip if already have position in this coin
        coin_positions = [p for p in portfolio['positions'].values() if p['coin'] == coin]
        if len(coin_positions) >= 1:  # Max 1 position per coin
            continue
            
        trade_params = ai_determine_trade_params(
            coin, market_data, sentiment_scores[coin], price_histories, learning_data
        )
        
        if trade_params and trade_params['direction'] != 'SKIP':
            trading_opportunities[coin] = trade_params
    
    return trading_opportunities, sentiment_scores

def run_trading_bot():
    """Advanced leverage trading bot loop"""
    price_histories = {'BTC': [], 'ETH': [], 'SOL': [], 'BNB': []}
    
    print("🚀 AI LEVERAGE TRADING BOT STARTED")
    print("💰 Starting Balance: $1,000")
    print("⚡ Leverage: 10-20x based on AI confidence")
    print("🧠 AI Learning: Adapting from every trade")
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