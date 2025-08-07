import requests
import time
import os
from openai import OpenAI
import statistics

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

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

def analyze_order_book(order_book):
    """Analyze order book for support/resistance and liquidity"""
    bids = [[float(bid[0]), float(bid[1])] for bid in order_book['bids'][:10]]
    asks = [[float(ask[0]), float(ask[1])] for ask in order_book['asks'][:10]]
    
    # Calculate liquidity metrics
    bid_liquidity = sum([price * volume for price, volume in bids])
    ask_liquidity = sum([price * volume for price, volume in asks])
    total_liquidity = bid_liquidity + ask_liquidity
    
    # Liquidity imbalance
    imbalance = (bid_liquidity - ask_liquidity) / total_liquidity if total_liquidity > 0 else 0
    
    # Spread analysis
    best_bid = bids[0][0] if bids else 0
    best_ask = asks[0][0] if asks else 0
    spread = ((best_ask - best_bid) / best_bid * 100) if best_bid > 0 else 0
    
    # Find large walls (support/resistance)
    large_bid_walls = [bid for bid in bids if bid[1] > statistics.mean([b[1] for b in bids]) * 2]
    large_ask_walls = [ask for ask in asks if ask[1] > statistics.mean([a[1] for a in asks]) * 2]
    
    return {
        'liquidity_imbalance': round(imbalance, 3),
        'spread_percent': round(spread, 4),
        'bid_liquidity': round(bid_liquidity, 2),
        'ask_liquidity': round(ask_liquidity, 2),
        'large_bid_walls': len(large_bid_walls),
        'large_ask_walls': len(large_ask_walls),
        'best_bid': best_bid,
        'best_ask': best_ask
    }

def analyze_volume_profile(trades, current_price):
    """Analyze volume distribution around price levels"""
    if not trades:
        return {'volume_weighted_price': current_price, 'volume_trend': 'neutral'}
    
    # Calculate volume weighted average price (VWAP)
    total_volume = 0
    volume_price_sum = 0
    
    buy_volume = 0
    sell_volume = 0
    
    for trade in trades[-50:]:  # Last 50 trades
        price = float(trade['price'])
        volume = float(trade['qty'])
        is_buyer = trade.get('isBuyerMaker', False)
        
        total_volume += volume
        volume_price_sum += price * volume
        
        if is_buyer:
            sell_volume += volume  # Buyer maker means sell order filled
        else:
            buy_volume += volume   # Seller maker means buy order filled
    
    vwap = volume_price_sum / total_volume if total_volume > 0 else current_price
    
    # Volume trend analysis
    volume_ratio = buy_volume / (buy_volume + sell_volume) if (buy_volume + sell_volume) > 0 else 0.5
    
    if volume_ratio > 0.6:
        volume_trend = 'bullish'
    elif volume_ratio < 0.4:
        volume_trend = 'bearish'
    else:
        volume_trend = 'neutral'
    
    return {
        'vwap': round(vwap, 2),
        'volume_trend': volume_trend,
        'buy_volume_ratio': round(volume_ratio, 3)
    }

def detect_supply_demand_zones(prices, volumes=None):
    """Identify key supply and demand zones"""
    if len(prices) < 10:
        return {'support_levels': [], 'resistance_levels': []}
    
    recent_prices = prices[-20:]  # Last 20 data points
    current_price = recent_prices[-1]
    
    # Find local support (demand zones)
    support_levels = []
    resistance_levels = []
    
    for i in range(2, len(recent_prices) - 2):
        # Support: price is lower than surrounding prices
        if (recent_prices[i] < recent_prices[i-1] and 
            recent_prices[i] < recent_prices[i-2] and
            recent_prices[i] < recent_prices[i+1] and 
            recent_prices[i] < recent_prices[i+2]):
            support_levels.append(recent_prices[i])
        
        # Resistance: price is higher than surrounding prices
        if (recent_prices[i] > recent_prices[i-1] and 
            recent_prices[i] > recent_prices[i-2] and
            recent_prices[i] > recent_prices[i+1] and 
            recent_prices[i] > recent_prices[i+2]):
            resistance_levels.append(recent_prices[i])
    
    # Filter to most relevant levels (within 5% of current price)
    relevant_support = [s for s in support_levels if abs(s - current_price) / current_price < 0.05]
    relevant_resistance = [r for r in resistance_levels if abs(r - current_price) / current_price < 0.05]
    
    return {
        'support_levels': relevant_support[-3:],  # Keep 3 most recent
        'resistance_levels': relevant_resistance[-3:]
    }

def comprehensive_ai_analysis(market_data, price_histories, technical_data):
    """Advanced AI market analysis"""
    if not client:
        return {coin: "WAIT (No API key)" for coin in market_data.keys()}
    
    # Build comprehensive analysis prompt
    analysis_text = "COMPREHENSIVE CRYPTO MARKET ANALYSIS\n\n"
    
    for coin, data in market_data.items():
        tech = technical_data[coin]
        rsi = calculate_rsi(price_histories.get(coin, []))
        
        analysis_text += f"--- {coin} ANALYSIS ---\n"
        analysis_text += f"Price: ${data['price']:,.2f} ({data['change_24h']:+.2f}%)\n"
        analysis_text += f"24h Range: ${data['low_24h']:,.2f} - ${data['high_24h']:,.2f}\n"
        analysis_text += f"RSI: {rsi:.1f}\n"
        analysis_text += f"VWAP: ${tech['volume_profile']['vwap']:,.2f}\n"
        analysis_text += f"Volume Trend: {tech['volume_profile']['volume_trend']}\n"
        analysis_text += f"Buy/Sell Ratio: {tech['volume_profile']['buy_volume_ratio']:.2f}\n"
        analysis_text += f"Liquidity Imbalance: {tech['order_book']['liquidity_imbalance']:+.3f}\n"
        analysis_text += f"Spread: {tech['order_book']['spread_percent']:.3f}%\n"
        analysis_text += f"Support Levels: {tech['supply_demand']['support_levels']}\n"
        analysis_text += f"Resistance Levels: {tech['supply_demand']['resistance_levels']}\n\n"
    
    prompt = f"""{analysis_text}

TRADING DECISION FRAMEWORK:
- BUY Signals: RSI < 30, bullish volume trend, price near support, positive liquidity imbalance
- SELL Signals: RSI > 70, bearish volume trend, price near resistance, negative liquidity imbalance  
- WAIT Signals: Conflicting signals, high spread, unclear trend, RSI 30-70 with neutral volume

For each coin, analyze ALL factors and provide ONE decision: BUY, SELL, or WAIT
Format response as: BTC:DECISION ETH:DECISION SOL:DECISION BNB:DECISION
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

# Initialize data storage
price_histories = {'BTC': [], 'ETH': [], 'SOL': [], 'BNB': []}
volume_histories = {'BTC': [], 'ETH': [], 'SOL': [], 'BNB': []}

print("🚀 Advanced Crypto Trading Bot Started")
print("Analyzing: Order Books, Volume Profiles, RSI, Supply/Demand Zones")

while True:
    try:
        print("\n" + "="*80)
        print(f"📊 MARKET SCAN - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Get comprehensive market data
        market_data = get_market_data()
        
        # Update price histories
        for coin in market_data:
            price_histories[coin].append(market_data[coin]['price'])
            volume_histories[coin].append(market_data[coin]['volume'])
            
            # Keep last 50 data points
            if len(price_histories[coin]) > 50:
                price_histories[coin] = price_histories[coin][-50:]
                volume_histories[coin] = volume_histories[coin][-50:]
        
        # Perform technical analysis for each coin
        technical_data = {}
        for coin, data in market_data.items():
            order_book_analysis = analyze_order_book(data['order_book'])
            volume_analysis = analyze_volume_profile(data['recent_trades'], data['price'])
            supply_demand = detect_supply_demand_zones(price_histories[coin], volume_histories[coin])
            
            technical_data[coin] = {
                'order_book': order_book_analysis,
                'volume_profile': volume_analysis,
                'supply_demand': supply_demand
            }
            
            # Display analysis
            rsi = calculate_rsi(price_histories[coin])
            print(f"\n💰 {coin}: ${data['price']:,.2f} ({data['change_24h']:+.2f}%)")
            print(f"   RSI: {rsi:.1f} | VWAP: ${volume_analysis['vwap']:,.2f} | Volume: {volume_analysis['volume_trend']}")
            print(f"   Spread: {order_book_analysis['spread_percent']:.3f}% | Liquidity: {order_book_analysis['liquidity_imbalance']:+.3f}")
            print(f"   Support: {len(supply_demand['support_levels'])} levels | Resistance: {len(supply_demand['resistance_levels'])} levels")
        
        # Get AI analysis (after sufficient data)
        if all(len(history) >= 10 for history in price_histories.values()):
            print(f"\n🤖 AI ANALYSIS:")
            decisions = comprehensive_ai_analysis(market_data, price_histories, technical_data)
            
            for coin, decision in decisions.items():
                indicator = "🟢" if decision == "BUY" else "🔴" if decision == "SELL" else "🟡"
                print(f"   {indicator} {coin}: {decision}")
        else:
            print(f"\n📈 Building analysis database... ({min(len(h) for h in price_histories.values())}/10 data points)")
        
        print("="*80)
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    time.sleep(90)  # Analyze every 90 seconds