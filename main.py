import requests
import time
import os
from openai import OpenAI
import statistics
import json
from datetime import datetime
from flask import Flask, jsonify, send_from_directory
import threading

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

# Paper Trading Portfolio
portfolio = {
    'balance': 1000.0,  # Starting with $1000 virtual money
    'positions': {},    # {'BTC': {'amount': 0.01, 'entry_price': 45000, 'entry_time': '...'}}
    'trade_history': []
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
        # Get current market data
        current_market_data = get_market_data()
        
        # Calculate portfolio value
        total_value = calculate_portfolio_value(current_market_data)
        
        # Format response data
        response_data = {
            'total_value': total_value,
            'balance': portfolio['balance'],
            'positions': portfolio['positions'],
            'trade_history': portfolio['trade_history'],
            'market_data': current_market_data,
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
        
        # Recent trades for volume analysis
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

def execute_paper_trade(coin, decision, current_price, analysis_data):
    """Execute paper trading based on AI decision"""
    global portfolio
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if decision == "BUY" and coin not in portfolio['positions']:
        # Calculate position size (risk 2% of portfolio per trade)
        risk_amount = portfolio['balance'] * 0.02
        position_size = risk_amount / current_price
        
        if portfolio['balance'] >= risk_amount:
            # Open position
            portfolio['positions'][coin] = {
                'amount': position_size,
                'entry_price': current_price,
                'entry_time': timestamp,
                'stop_loss': current_price * 0.95,  # 5% stop loss
                'take_profit': current_price * 1.10  # 10% take profit
            }
            
            portfolio['balance'] -= risk_amount
            
            trade_record = {
                'time': timestamp,
                'coin': coin,
                'action': 'BUY',
                'price': current_price,
                'amount': position_size,
                'value': risk_amount,
                'reason': f"AI Signal + RSI: {analysis_data.get('rsi', 'N/A')}"
            }
            portfolio['trade_history'].append(trade_record)
            
            print(f"   📈 BOUGHT {coin}: ${risk_amount:.2f} at ${current_price:,.2f}")
            print(f"      Amount: {position_size:.6f} {coin}")
            print(f"      Stop Loss: ${portfolio['positions'][coin]['stop_loss']:,.2f}")
            print(f"      Take Profit: ${portfolio['positions'][coin]['take_profit']:,.2f}")
    
    elif decision == "SELL" and coin in portfolio['positions']:
        # Close position
        position = portfolio['positions'][coin]
        current_value = position['amount'] * current_price
        profit_loss = current_value - (position['amount'] * position['entry_price'])
        
        portfolio['balance'] += current_value
        
        trade_record = {
            'time': timestamp,
            'coin': coin,
            'action': 'SELL',
            'price': current_price,
            'amount': position['amount'],
            'value': current_value,
            'profit_loss': profit_loss,
            'reason': f"AI Signal + RSI: {analysis_data.get('rsi', 'N/A')}"
        }
        portfolio['trade_history'].append(trade_record)
        
        del portfolio['positions'][coin]
        
        profit_emoji = "💰" if profit_loss > 0 else "📉"
        print(f"   {profit_emoji} SOLD {coin}: ${current_value:.2f} at ${current_price:,.2f}")
        print(f"      P&L: ${profit_loss:+.2f} ({(profit_loss/current_value)*100:+.2f}%)")

def check_stop_loss_take_profit(coin, current_price):
    """Check if stop loss or take profit should trigger"""
    global portfolio
    
    if coin not in portfolio['positions']:
        return
    
    position = portfolio['positions'][coin]
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Check stop loss
    if current_price <= position['stop_loss']:
        current_value = position['amount'] * current_price
        profit_loss = current_value - (position['amount'] * position['entry_price'])
        
        portfolio['balance'] += current_value
        
        trade_record = {
            'time': timestamp,
            'coin': coin,
            'action': 'STOP_LOSS',
            'price': current_price,
            'amount': position['amount'],
            'value': current_value,
            'profit_loss': profit_loss,
            'reason': f"Stop Loss triggered at ${current_price:,.2f}"
        }
        portfolio['trade_history'].append(trade_record)
        
        del portfolio['positions'][coin]
        print(f"   🛑 STOP LOSS {coin}: ${current_value:.2f} (P&L: ${profit_loss:+.2f})")
    
    # Check take profit
    elif current_price >= position['take_profit']:
        current_value = position['amount'] * current_price
        profit_loss = current_value - (position['amount'] * position['entry_price'])
        
        portfolio['balance'] += current_value
        
        trade_record = {
            'time': timestamp,
            'coin': coin,
            'action': 'TAKE_PROFIT',
            'price': current_price,
            'amount': position['amount'],
            'value': current_value,
            'profit_loss': profit_loss,
            'reason': f"Take Profit triggered at ${current_price:,.2f}"
        }
        portfolio['trade_history'].append(trade_record)
        
        del portfolio['positions'][coin]
        print(f"   🎯 TAKE PROFIT {coin}: ${current_value:.2f} (P&L: ${profit_loss:+.2f})")

def calculate_portfolio_value(market_data):
    """Calculate total portfolio value including open positions"""
    total_value = portfolio['balance']
    
    for coin, position in portfolio['positions'].items():
        if coin in market_data:
            current_value = position['amount'] * market_data[coin]['price']
            total_value += current_value
    
    return total_value

def display_portfolio_status(market_data):
    """Display current portfolio status"""
    total_value = calculate_portfolio_value(market_data)
    profit_loss = total_value - 1000  # Starting balance was $1000
    
    print(f"\n💼 PORTFOLIO STATUS:")
    print(f"   Cash Balance: ${portfolio['balance']:.2f}")
    print(f"   Total Value: ${total_value:.2f}")
    print(f"   P&L: ${profit_loss:+.2f} ({(profit_loss/1000)*100:+.2f}%)")
    print(f"   Total Trades: {len(portfolio['trade_history'])}")
    
    if portfolio['positions']:
        print(f"   📊 Open Positions:")
        for coin, position in portfolio['positions'].items():
            current_price = market_data[coin]['price']
            current_value = position['amount'] * current_price
            unrealized_pnl = current_value - (position['amount'] * position['entry_price'])
            
            print(f"      {coin}: {position['amount']:.6f} @ ${position['entry_price']:,.2f}")
            print(f"           Current: ${current_price:,.2f} | Unrealized P&L: ${unrealized_pnl:+.2f}")

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

def comprehensive_ai_analysis(market_data, price_histories, technical_data):
    """AI analysis with trading context"""
    if not client:
        return {coin: "WAIT" for coin in market_data.keys()}
    
    # Include portfolio context in AI decision
    portfolio_context = f"Current Portfolio Value: ${calculate_portfolio_value(market_data):.2f}\n"
    portfolio_context += f"Open Positions: {list(portfolio['positions'].keys())}\n"
    
    analysis_text = f"{portfolio_context}\nMARKET ANALYSIS:\n\n"
    
    for coin, data in market_data.items():
        rsi = calculate_rsi(price_histories.get(coin, []))
        
        analysis_text += f"{coin}: ${data['price']:,.2f} ({data['change_24h']:+.2f}%)\n"
        analysis_text += f"RSI: {rsi:.1f} | Volume: {data['volume']:,.0f}\n"
        
        if coin in portfolio['positions']:
            pos = portfolio['positions'][coin]
            unrealized = (data['price'] - pos['entry_price']) * pos['amount']
            analysis_text += f"OPEN POSITION: Entry ${pos['entry_price']:,.2f} | P&L: ${unrealized:+.2f}\n"
        
        analysis_text += "\n"
    
    prompt = f"""{analysis_text}

TRADING RULES:
- Only BUY if strong bullish signals and NO existing position
- Only SELL if bearish signals and HAVE existing position  
- Use WAIT for unclear signals or portfolio management
- Consider RSI: BUY if <30, SELL if >70, WAIT if 30-70
- Maximum 2 open positions at once

For each coin, provide: BUY, SELL, or WAIT
Format: BTC:DECISION ETH:DECISION SOL:DECISION BNB:DECISION
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        
        ai_text = response.choices[0].message.content.strip()
        decisions = {}
        
        for coin in market_data.keys():
            if f"{coin}:" in ai_text:
                decision = ai_text.split(f"{coin}:")[1].split()[0]
                decisions[coin] = decision.upper()
            else:
                decisions[coin] = "WAIT"
        
        return decisions
        
    except Exception as e:
        print(f"AI Error: {e}")
        return {coin: "WAIT" for coin in market_data.keys()}

def run_trading_bot():
    """Main trading bot loop"""
    # Initialize data storage
    price_histories = {'BTC': [], 'ETH': [], 'SOL': [], 'BNB': []}
    
    print("🚀 PAPER TRADING BOT STARTED")
    print("💰 Starting Balance: $1,000")
    print("📊 Risk per trade: 2% of portfolio")
    print("🌐 Dashboard available at: http://localhost:5000")
    
    while True:
        try:
            print("\n" + "="*80)
            print(f"📊 TRADING SCAN - {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)
            
            # Get market data
            market_data = get_market_data()
            
            # Update price histories
            for coin in market_data:
                price_histories[coin].append(market_data[coin]['price'])
                if len(price_histories[coin]) > 50:
                    price_histories[coin] = price_histories[coin][-50:]
            
            # Check stop losses and take profits
            for coin in list(portfolio['positions'].keys()):
                check_stop_loss_take_profit(coin, market_data[coin]['price'])
            
            # Display current prices and analysis
            for coin, data in market_data.items():
                rsi = calculate_rsi(price_histories[coin])
                print(f"💰 {coin}: ${data['price']:,.2f} ({data['change_24h']:+.2f}%) | RSI: {rsi:.1f}")
            
            # Get AI decisions and execute trades
            if all(len(history) >= 10 for history in price_histories.values()):
                print(f"\n🤖 AI TRADING DECISIONS:")
                decisions = comprehensive_ai_analysis(market_data, price_histories, {})
                
                for coin, decision in decisions.items():
                    analysis_data = {'rsi': calculate_rsi(price_histories[coin])}
                    current_price = market_data[coin]['price']
                    
                    indicator = "🟢" if decision == "BUY" else "🔴" if decision == "SELL" else "🟡"
                    print(f"   {indicator} {coin}: {decision}")
                    
                    # Execute the trade
                    execute_paper_trade(coin, decision, current_price, analysis_data)
            else:
                data_count = min(len(h) for h in price_histories.values())
                print(f"\n📈 Building analysis database... ({data_count}/10 data points)")
            
            # Display portfolio status
            display_portfolio_status(market_data)
            
            print("="*80)
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        time.sleep(120)  # Trade analysis every 2 minutes

# Start both Flask web server and trading bot
if __name__ == "__main__":
    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=run_flask_app, daemon=True)
    flask_thread.start()
    
    # Start the trading bot in main thread
    run_trading_bot()