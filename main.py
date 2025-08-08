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
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

# External data API keys
NEWS_API_KEY = os.getenv("NEWS_API_KEY")  # Get from newsapi.org
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")  # Get from alphavantage.co

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
FULL_ANALYSIS_INTERVAL = 60  # Full market analysis every 1 minute
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

# ========== MEMORY SYSTEMS ==========

class TradingMemory:
    """Maintains context across API calls for better decision making"""
    def __init__(self):
        self.conversation_history = [
            {"role": "system", "content": """You are an evolving trading AI that learns from every decision. 
             Remember patterns, outcomes, and market conditions to make increasingly better trades.
             Build on your experience and adapt your strategy based on what works."""}
        ]
        self.recent_context = deque(maxlen=100)  # Last 100 decisions
        self.market_patterns = {}
        self.successful_strategies = []
        self.max_context_length = 12000  # Token limit
        
    def add_decision(self, coin, analysis, decision, outcome=None):
        """Add a trading decision to memory"""
        context_entry = {
            'coin': coin,
            'analysis': analysis,
            'decision': decision,
            'timestamp': datetime.now(),
            'outcome': outcome
        }
        self.recent_context.append(context_entry)
        
        # Add to conversation history
        self.conversation_history.append({
            "role": "user", 
            "content": f"ANALYSIS for {coin}: {analysis}"
        })
        
        self.conversation_history.append({
            "role": "assistant",
            "content": f"DECISION: {decision}"
        })
        
        self._trim_context()
    
    def add_trade_outcome(self, coin, outcome_details):
        """Record trade outcome for learning"""
        self.conversation_history.append({
            "role": "user",
            "content": f"OUTCOME for {coin}: {outcome_details}. Learn from this for future {coin} decisions."
        })
        
        # Update recent context with outcome
        for entry in reversed(self.recent_context):
            if entry['coin'] == coin and entry['outcome'] is None:
                entry['outcome'] = outcome_details
                break
        
        self._trim_context()
    
    def get_context_for_coin(self, coin):
        """Get relevant memory context for a specific coin"""
        # Get recent decisions for this coin
        coin_history = [entry for entry in self.recent_context 
                       if entry['coin'] == coin][-5:]  # Last 5 decisions
        
        # Get recent market patterns
        recent_patterns = list(self.recent_context)[-20:]  # Last 20 overall
        
        context = f"""
TRADING MEMORY for {coin}:

Recent {coin} decisions:
{self._format_coin_history(coin_history)}

Recent market patterns:
{self._format_recent_patterns(recent_patterns)}

Successful strategies learned:
{self._get_successful_patterns()}

Market insights gained:
{self._get_market_insights()}
"""
        return context
    
    def get_decision_with_memory(self, coin, current_analysis):
        """Make AI decision with full memory context"""
        if not client:
            return None
            
        # Get memory context
        memory_context = self.get_context_for_coin(coin)
        
        # Combine memory with current analysis
        full_prompt = f"""
{memory_context}

CURRENT ANALYSIS for {coin}:
{current_analysis}

Based on your memory, learned patterns, and current analysis, make your decision.
Consider what worked before and what failed for {coin} specifically.
"""
        
        # Add to conversation
        messages = self.conversation_history + [{
            "role": "user",
            "content": full_prompt
        }]
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=200,
                temperature=0.4
            )
            
            track_cost('openai', 0.00012, '400')
            
            decision = response.choices[0].message.content
            self.add_decision(coin, current_analysis, decision)
            
            return decision
            
        except Exception as e:
            print(f"Memory AI Error: {e}")
            return None
    
    def _format_coin_history(self, coin_history):
        """Format coin-specific history for prompt"""
        if not coin_history:
            return "No recent history for this coin."
        
        formatted = []
        for entry in coin_history:
            outcome_str = f" → {entry['outcome']}" if entry['outcome'] else " → Pending"
            formatted.append(f"• {entry['decision'][:100]}{outcome_str}")
        
        return "\n".join(formatted)
    
    def _format_recent_patterns(self, patterns):
        """Format recent market patterns"""
        if not patterns:
            return "No recent patterns."
        
        formatted = []
        for entry in patterns[-10:]:  # Last 10
            outcome_str = "✅" if entry['outcome'] and "profit" in str(entry['outcome']).lower() else "❌" if entry['outcome'] else "⏳"
            formatted.append(f"{outcome_str} {entry['coin']}: {entry['decision'][:50]}...")
        
        return "\n".join(formatted)
    
    def _get_successful_patterns(self):
        """Identify successful patterns from memory"""
        successful = []
        for entry in self.recent_context:
            if entry['outcome'] and any(word in str(entry['outcome']).lower() 
                                      for word in ['profit', 'successful', 'win']):
                successful.append(f"✅ {entry['coin']}: {entry['decision'][:80]}...")
        
        return "\n".join(successful[-5:]) if successful else "Building pattern database..."
    
    def _get_market_insights(self):
        """Extract market insights from memory"""
        insights = []
        
        # Analyze coin performance
        coin_performance = {}
        for entry in self.recent_context:
            if entry['outcome']:
                coin = entry['coin']
                if coin not in coin_performance:
                    coin_performance[coin] = {'wins': 0, 'losses': 0}
                
                if any(word in str(entry['outcome']).lower() 
                      for word in ['profit', 'successful']):
                    coin_performance[coin]['wins'] += 1
                else:
                    coin_performance[coin]['losses'] += 1
        
        # Best performing coins
        best_coins = []
        for coin, perf in coin_performance.items():
            total = perf['wins'] + perf['losses']
            if total >= 3:  # Minimum sample size
                win_rate = perf['wins'] / total
                if win_rate > 0.6:
                    best_coins.append(f"{coin}: {win_rate:.0%} win rate")
        
        if best_coins:
            insights.append(f"Best performers: {', '.join(best_coins[:3])}")
        
        return "\n".join(insights) if insights else "Gathering market intelligence..."
    
    def _trim_context(self):
        """Keep conversation within token limits"""
        if len(str(self.conversation_history)) > self.max_context_length:
            # Keep system message and recent exchanges
            self.conversation_history = [
                self.conversation_history[0]  # System message
            ] + self.conversation_history[-30:]  # Last 30 messages

class PatternLearner:
    """Advanced pattern recognition and learning system"""
    def __init__(self):
        self.patterns = {}  # pattern_hash -> pattern_data
        self.embeddings = []  # Trade embeddings for similarity
        self.outcomes = []   # Corresponding outcomes
        self.metadata = []   # Trade metadata
        
    def create_pattern_hash(self, market_conditions, indicators):
        """Create unique identifier for market conditions"""
        regime = market_conditions.get('regime', 'UNKNOWN')
        rsi_range = 'LOW' if indicators.get('rsi', 50) < 40 else 'HIGH' if indicators.get('rsi', 50) > 60 else 'MID'
        volume_trend = market_conditions.get('volume_trend', 'STABLE')
        tf_alignment = market_conditions.get('tf_alignment', 'NEUTRAL')
        
        pattern = f"{regime}_{rsi_range}_{volume_trend}_{tf_alignment}"
        return pattern
    
    def create_trade_embedding(self, market_data, indicators, external_data):
        """Convert trade context to numerical vector for similarity matching"""
        features = [
            market_data.get('price', 0),
            market_data.get('change_24h', 0),
            indicators.get('rsi', 50),
            indicators.get('atr', 0),
            indicators.get('volume_ratio', 1),
            indicators.get('bb_width', 0),
            external_data.get('news_sentiment', 0),
            external_data.get('fear_greed', 50),
            external_data.get('social_sentiment', 0),
            len(indicators.get('support_levels', [])),
            len(indicators.get('resistance_levels', [])),
        ]
        
        # Normalize features to prevent scale issues
        features = [(f - min(features)) / (max(features) - min(features) + 0.001) 
                   for f in features]
        
        return np.array(features)
    
    def record_pattern_outcome(self, market_conditions, indicators, external_data, 
                             decision, was_successful, pnl_percent, coin):
        """Record pattern outcome for learning"""
        pattern_hash = self.create_pattern_hash(market_conditions, indicators)
        
        if pattern_hash not in self.patterns:
            self.patterns[pattern_hash] = {
                'successes': 0,
                'failures': 0,
                'total_trades': 0,
                'avg_return': 0,
                'conditions': market_conditions,
                'best_decision': None,
                'coins_traded': set()
            }
        
        pattern = self.patterns[pattern_hash]
        pattern['total_trades'] += 1
        pattern['coins_traded'].add(coin)
        
        # Update returns
        old_avg = pattern['avg_return']
        pattern['avg_return'] = (old_avg * (pattern['total_trades'] - 1) + pnl_percent) / pattern['total_trades']
        
        if was_successful:
            pattern['successes'] += 1
            pattern['best_decision'] = decision
        else:
            pattern['failures'] += 1
        
        # Store embedding for similarity matching
        embedding = self.create_trade_embedding(
            market_conditions.get('market_data', {}), 
            indicators, 
            external_data
        )
        
        self.embeddings.append(embedding)
        self.outcomes.append(1 if was_successful else 0)
        self.metadata.append({
            'coin': coin,
            'decision': decision,
            'pnl_percent': pnl_percent,
            'pattern_hash': pattern_hash,
            'timestamp': datetime.now()
        })
        
        # Keep only recent embeddings (memory management)
        if len(self.embeddings) > 1000:
            self.embeddings = self.embeddings[-500:]
            self.outcomes = self.outcomes[-500:]
            self.metadata = self.metadata[-500:]
    
    def get_pattern_recommendation(self, market_conditions, indicators, external_data, coin):
        """Get recommendation based on learned patterns"""
        pattern_hash = self.create_pattern_hash(market_conditions, indicators)
        
        recommendations = {}
        
        # Check exact pattern match
        if pattern_hash in self.patterns:
            pattern = self.patterns[pattern_hash]
            
            if pattern['total_trades'] >= 3:  # Minimum sample size
                confidence = pattern['successes'] / pattern['total_trades']
                recommendations['exact_pattern'] = {
                    'action': pattern['best_decision'],
                    'confidence': confidence,
                    'sample_size': pattern['total_trades'],
                    'avg_return': pattern['avg_return'],
                    'reason': f"Exact pattern match: {confidence:.0%} success rate"
                }
        
        # Check similar patterns using embeddings
        if self.embeddings:
            current_embedding = self.create_trade_embedding(
                market_conditions.get('market_data', {}),
                indicators,
                external_data
            )
            
            similarities = cosine_similarity([current_embedding], self.embeddings)[0]
            
            # Find top 5 most similar trades
            top_indices = np.argsort(similarities)[-5:]
            similar_trades = []
            
            for i in top_indices:
                if similarities[i] > 0.7:  # High similarity threshold
                    similar_trades.append({
                        'similarity': similarities[i],
                        'outcome': self.outcomes[i],
                        'metadata': self.metadata[i]
                    })
            
            if similar_trades:
                success_rate = sum(trade['outcome'] * trade['similarity'] 
                                 for trade in similar_trades) / sum(trade['similarity'] for trade in similar_trades)
                
                recommendations['similar_patterns'] = {
                    'confidence': success_rate,
                    'sample_size': len(similar_trades),
                    'avg_similarity': np.mean([t['similarity'] for t in similar_trades]),
                    'reason': f"Similar pattern analysis: {success_rate:.0%} weighted success rate"
                }
        
        # Coin-specific recommendations
        coin_patterns = [p for p in self.patterns.values() if coin in p['coins_traded']]
        if coin_patterns:
            coin_successes = sum(p['successes'] for p in coin_patterns)
            coin_total = sum(p['total_trades'] for p in coin_patterns)
            
            if coin_total >= 5:
                coin_confidence = coin_successes / coin_total
                recommendations['coin_specific'] = {
                    'confidence': coin_confidence,
                    'sample_size': coin_total,
                    'reason': f"{coin} historical performance: {coin_confidence:.0%} success rate"
                }
        
        return recommendations

class ExternalDataManager:
    """Manages external data sources for market sentiment and news"""
    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
    def get_news_sentiment(self, coin):
        """Get news sentiment for a coin using News API"""
        cache_key = f"news_{coin}"
        
        if self._is_cached(cache_key):
            return self.cache[cache_key]['data']
        
        if not NEWS_API_KEY:
            return 0  # Neutral if no API key
        
        try:
            # Search for recent news about the coin
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f'{coin} OR {coin}USD OR {coin}USDT',
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 20,
                'from': (datetime.now() - timedelta(hours=24)).isoformat(),
                'apiKey': NEWS_API_KEY
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                articles = response.json().get('articles', [])
                
                if not articles:
                    sentiment = 0
                else:
                    # Analyze sentiment using OpenAI
                    headlines = [article['title'] for article in articles[:10]]
                    sentiment = self._analyze_news_sentiment(headlines, coin)
                
                self._cache_data(cache_key, sentiment)
                return sentiment
            
        except Exception as e:
            print(f"News API error for {coin}: {e}")
        
        return 0
    
    def get_fear_greed_index(self):
        """Get crypto fear and greed index"""
        cache_key = "fear_greed"
        
        if self._is_cached(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                fear_greed = int(data['data'][0]['value'])
                
                self._cache_data(cache_key, fear_greed)
                return fear_greed
                
        except Exception as e:
            print(f"Fear & Greed API error: {e}")
        
        return 50  # Neutral default
    
    def get_social_sentiment(self, coin):
        """Get social media sentiment (simplified using news as proxy)"""
        # For now, use news sentiment as social sentiment proxy
        # In production, you could integrate Twitter API, Reddit API, etc.
        news_sentiment = self.get_news_sentiment(coin)
        
        # Add some randomness to differentiate from news
        social_modifier = hash(coin + str(datetime.now().hour)) % 21 - 10  # -10 to +10
        social_sentiment = max(-100, min(100, news_sentiment + social_modifier))
        
        return social_sentiment
    
    def get_whale_activity(self, coin):
        """Get whale movement data (mock implementation)"""
        # This would integrate with Whale Alert API in production
        # For now, return neutral
        return {'large_transactions': 0, 'net_flow': 0}
    
    def get_comprehensive_sentiment(self, coin):
        """Get all external sentiment data for a coin"""
        return {
            'news_sentiment': self.get_news_sentiment(coin),
            'fear_greed': self.get_fear_greed_index(),
            'social_sentiment': self.get_social_sentiment(coin),
            'whale_activity': self.get_whale_activity(coin)
        }
    
    def _analyze_news_sentiment(self, headlines, coin):
        """Analyze news sentiment using AI"""
        if not client or not headlines:
            return 0
        
        try:
            headlines_text = '\n'.join(headlines)
            
            prompt = f"""
Analyze the sentiment of these recent news headlines about {coin}:

{headlines_text}

Rate the overall sentiment from -100 (very negative) to +100 (very positive).
Consider price implications, adoption news, regulatory changes, etc.

Return only a number between -100 and +100.
"""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.3
            )
            
            track_cost('openai', 0.00005, '100')
            
            # Extract number from response
            sentiment_text = response.choices[0].message.content.strip()
            sentiment = int(''.join(filter(lambda x: x.isdigit() or x == '-', sentiment_text)))
            
            return max(-100, min(100, sentiment))
            
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return 0
    
    def _is_cached(self, key):
        """Check if data is cached and still valid"""
        if key not in self.cache:
            return False
        
        cache_time = self.cache[key]['timestamp']
        return (datetime.now() - cache_time).seconds < self.cache_duration
    
    def _cache_data(self, key, data):
        """Cache data with timestamp"""
        self.cache[key] = {
            'data': data,
            'timestamp': datetime.now()
        }

class SmartProfitManager:
    """Advanced profit-taking and stop management system"""
    
    def __init__(self):
        self.profit_signals = {}
        self.exit_patterns = {}
        
    def analyze_profit_exit_signals(self, position, current_price, market_data, multi_tf_data, external_data):
        """Comprehensive profit-taking analysis"""
        
        coin = position['coin']
        entry_price = position['entry_price']
        direction = position['direction']
        
        # Calculate current profit
        if direction == 'LONG':
            profit_pct = (current_price - entry_price) / entry_price * 100
        else:
            profit_pct = (entry_price - current_price) / entry_price * 100
        
        # Only analyze if in profit
        if profit_pct <= 0:
            return {'action': 'HOLD', 'reason': 'Not in profit', 'signals': []}
        
        exit_signals = []
        signal_strength = 0
        
        # 1. RSI Divergence Analysis
        rsi_signal = self._analyze_rsi_divergence(multi_tf_data, current_price, direction)
        if rsi_signal['strength'] > 0:
            exit_signals.append(rsi_signal)
            signal_strength += rsi_signal['strength']
        
        # 2. Volume Analysis
        volume_signal = self._analyze_volume_patterns(market_data, multi_tf_data, direction)
        if volume_signal['strength'] > 0:
            exit_signals.append(volume_signal)
            signal_strength += volume_signal['strength']
        
        # 3. Support/Resistance Interaction
        sr_signal = self._analyze_support_resistance_interaction(
            current_price, multi_tf_data, direction, profit_pct
        )
        if sr_signal['strength'] > 0:
            exit_signals.append(sr_signal)
            signal_strength += sr_signal['strength']
        
        # 4. Momentum Shift Detection
        momentum_signal = self._analyze_momentum_shift(multi_tf_data, direction)
        if momentum_signal['strength'] > 0:
            exit_signals.append(momentum_signal)
            signal_strength += momentum_signal['strength']
        
        # 5. Liquidity Analysis
        liquidity_signal = self._analyze_liquidity_conditions(market_data, multi_tf_data)
        if liquidity_signal['strength'] > 0:
            exit_signals.append(liquidity_signal)
            signal_strength += liquidity_signal['strength']
        
        # 6. Supply/Demand Zones
        supply_demand_signal = self._analyze_supply_demand_zones(
            current_price, multi_tf_data, direction
        )
        if supply_demand_signal['strength'] > 0:
            exit_signals.append(supply_demand_signal)
            signal_strength += supply_demand_signal['strength']
        
        # 7. External Sentiment Shift
        sentiment_signal = self._analyze_sentiment_shift(
            external_data, position.get('external_data', {}), direction
        )
        if sentiment_signal['strength'] > 0:
            exit_signals.append(sentiment_signal)
            signal_strength += sentiment_signal['strength']
        
        # 8. Time-based Profit Taking
        time_signal = self._analyze_time_based_exit(position, profit_pct)
        if time_signal['strength'] > 0:
            exit_signals.append(time_signal)
            signal_strength += time_signal['strength']
        
        # Decision Logic
        action = self._determine_profit_action(signal_strength, profit_pct, exit_signals)
        
        return {
            'action': action['type'],
            'reason': action['reason'], 
            'signals': exit_signals,
            'signal_strength': signal_strength,
            'profit_pct': profit_pct,
            'new_stop_loss': action.get('new_stop_loss'),
            'partial_exit_pct': action.get('partial_exit_pct')
        }
    
    def _analyze_rsi_divergence(self, multi_tf_data, current_price, direction):
        """Detect RSI divergence for profit taking"""
        if '1h' not in multi_tf_data or '15m' not in multi_tf_data:
            return {'strength': 0, 'signal': 'No TF data'}
        
        # Check 1H and 15M RSI
        h1_closes = multi_tf_data['1h']['closes'][-20:]
        m15_closes = multi_tf_data['15m']['closes'][-20:]
        
        if len(h1_closes) < 15 or len(m15_closes) < 15:
            return {'strength': 0, 'signal': 'Insufficient data'}
        
        h1_rsi = calculate_rsi(h1_closes)
        m15_rsi = calculate_rsi(m15_closes)
        
        # Look for divergence
        recent_highs_h1 = h1_closes[-5:]
        recent_highs_m15 = m15_closes[-5:]
        
        price_making_higher_highs = max(recent_highs_h1) > max(h1_closes[-10:-5])
        price_making_lower_lows = min(recent_highs_h1) < min(h1_closes[-10:-5])
        
        divergence_strength = 0
        signal_type = ""
        
        if direction == 'LONG':
            # Bearish divergence: Price higher highs, RSI lower highs
            if price_making_higher_highs and h1_rsi < 65:
                if m15_rsi > 70:  # Overbought on shorter TF
                    divergence_strength = 3
                    signal_type = "Bearish RSI divergence detected"
                elif h1_rsi > 60:
                    divergence_strength = 2
                    signal_type = "Weak bearish RSI divergence"
        else:  # SHORT
            # Bullish divergence: Price lower lows, RSI higher lows
            if price_making_lower_lows and h1_rsi > 35:
                if m15_rsi < 30:  # Oversold on shorter TF
                    divergence_strength = 3
                    signal_type = "Bullish RSI divergence detected"
                elif h1_rsi < 40:
                    divergence_strength = 2
                    signal_type = "Weak bullish RSI divergence"
        
        return {
            'strength': divergence_strength,
            'signal': signal_type,
            'h1_rsi': h1_rsi,
            'm15_rsi': m15_rsi
        }
    
    def _analyze_volume_patterns(self, market_data, multi_tf_data, direction):
        """Analyze volume for profit-taking signals"""
        if '1h' not in multi_tf_data:
            return {'strength': 0, 'signal': 'No volume data'}
        
        volumes = multi_tf_data['1h']['volumes'][-20:]
        if len(volumes) < 10:
            return {'strength': 0, 'signal': 'Insufficient volume data'}
        
        # Current vs average volume
        current_volume = market_data['volume']
        avg_volume = sum(volumes[-10:]) / 10
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Volume trend analysis
        recent_volume_trend = sum(volumes[-5:]) / 5
        older_volume_trend = sum(volumes[-10:-5]) / 5
        volume_declining = recent_volume_trend < older_volume_trend * 0.8
        
        signal_strength = 0
        signal_type = ""
        
        # Volume exhaustion signals
        if volume_ratio > 2.0 and volume_declining:
            signal_strength = 3
            signal_type = "Volume exhaustion: High volume followed by decline"
        elif volume_declining and volume_ratio < 0.7:
            signal_strength = 2
            signal_type = "Low volume on continuation - momentum weakening"
        elif volume_ratio > 3.0:
            signal_strength = 1
            signal_type = "Abnormally high volume - potential climax"
        
        return {
            'strength': signal_strength,
            'signal': signal_type,
            'volume_ratio': volume_ratio,
            'volume_declining': volume_declining
        }
    
    def _analyze_support_resistance_interaction(self, current_price, multi_tf_data, direction, profit_pct):
        """Analyze interaction with key S/R levels"""
        if '1h' not in multi_tf_data:
            return {'strength': 0, 'signal': 'No price data'}
        
        # Get support/resistance levels
        sr_levels = identify_support_resistance(multi_tf_data['1h']['candles'])
        
        signal_strength = 0
        signal_type = ""
        
        if direction == 'LONG':
            # Check resistance levels above current price
            resistances_above = [r for r in sr_levels['resistance'] if r > current_price]
            
            if resistances_above:
                nearest_resistance = min(resistances_above)
                distance_to_resistance = (nearest_resistance - current_price) / current_price * 100
                
                if distance_to_resistance < 1.0:  # Within 1% of resistance
                    signal_strength = 4
                    signal_type = f"Approaching major resistance at ${nearest_resistance:.2f}"
                elif distance_to_resistance < 2.0 and profit_pct > 5:
                    signal_strength = 2
                    signal_type = f"Near resistance at ${nearest_resistance:.2f} with good profit"
        
        else:  # SHORT
            # Check support levels below current price
            supports_below = [s for s in sr_levels['support'] if s < current_price]
            
            if supports_below:
                nearest_support = max(supports_below)
                distance_to_support = (current_price - nearest_support) / current_price * 100
                
                if distance_to_support < 1.0:  # Within 1% of support
                    signal_strength = 4
                    signal_type = f"Approaching major support at ${nearest_support:.2f}"
                elif distance_to_support < 2.0 and profit_pct > 5:
                    signal_strength = 2
                    signal_type = f"Near support at ${nearest_support:.2f} with good profit"
        
        return {
            'strength': signal_strength,
            'signal': signal_type,
            'nearest_level': nearest_resistance if direction == 'LONG' else nearest_support if 'nearest_support' in locals() else None
        }
    
    def _analyze_momentum_shift(self, multi_tf_data, direction):
        """Detect momentum shifts using MACD and moving averages"""
        if '1h' not in multi_tf_data:
            return {'strength': 0, 'signal': 'No momentum data'}
        
        closes = multi_tf_data['1h']['closes']
        if len(closes) < 26:
            return {'strength': 0, 'signal': 'Insufficient data for MACD'}
        
        # Calculate MACD
        macd_data = calculate_macd(closes)
        
        # Moving average cross analysis
        ema_9 = calculate_ema(closes, 9)
        ema_21 = calculate_ema(closes, 21)
        
        signal_strength = 0
        signal_type = ""
        
        # MACD histogram turning
        histogram = macd_data['histogram']
        if abs(histogram) < 0.1:  # MACD histogram near zero
            signal_strength = 2
            signal_type = "MACD momentum weakening"
        
        # Moving average convergence
        ma_distance = abs(ema_9 - ema_21) / ema_21 * 100
        if ma_distance < 0.5:  # MAs converging
            signal_strength += 1
            signal_type += " | Moving averages converging"
        
        # Direction-specific signals
        if direction == 'LONG':
            if histogram < 0 and macd_data['macd'] > 0:  # MACD turning bearish
                signal_strength += 2
                signal_type += " | MACD turning bearish"
        else:  # SHORT
            if histogram > 0 and macd_data['macd'] < 0:  # MACD turning bullish
                signal_strength += 2
                signal_type += " | MACD turning bullish"
        
        return {
            'strength': min(signal_strength, 4),  # Cap at 4
            'signal': signal_type.strip(' |'),
            'macd_histogram': histogram,
            'ma_distance': ma_distance
        }
    
    def _analyze_liquidity_conditions(self, market_data, multi_tf_data):
        """Analyze liquidity conditions for optimal exit timing"""
        if '5m' not in multi_tf_data:
            return {'strength': 0, 'signal': 'No liquidity data'}
        
        # Volume profile analysis
        recent_closes = multi_tf_data['5m']['closes'][-20:]
        recent_volumes = multi_tf_data['5m']['volumes'][-20:]
        
        if len(recent_closes) < 10:
            return {'strength': 0, 'signal': 'Insufficient data'}
        
        # Calculate volume-weighted average price for recent period
        vwap = calculate_vwap(recent_closes, recent_volumes)
        current_price = market_data['price']
        
        # Bid-ask spread analysis (approximated from price action)
        recent_highs = multi_tf_data['5m']['highs'][-10:]
        recent_lows = multi_tf_data['5m']['lows'][-10:]
        
        avg_spread = sum(h - l for h, l in zip(recent_highs, recent_lows)) / len(recent_highs)
        spread_pct = avg_spread / current_price * 100
        
        signal_strength = 0
        signal_type = ""
        
        # High spread indicates low liquidity
        if spread_pct > 0.5:
            signal_strength = 2
            signal_type = "Low liquidity detected - consider exiting"
        
        # Price far from VWAP indicates potential reversion
        vwap_distance = abs(current_price - vwap) / vwap * 100
        if vwap_distance > 2.0:
            signal_strength += 1
            signal_type += " | Price extended from VWAP"
        
        return {
            'strength': signal_strength,
            'signal': signal_type.strip(' |'),
            'vwap_distance': vwap_distance,
            'spread_pct': spread_pct
        }
    
    def _analyze_supply_demand_zones(self, current_price, multi_tf_data, direction):
        """Identify supply/demand zones for profit taking"""
        if '4h' not in multi_tf_data:
            return {'strength': 0, 'signal': 'No zone data'}
        
        # Get 4H candles for zone analysis
        candles = multi_tf_data['4h']['candles'][-50:]  # Last 50 4H candles
        
        if len(candles) < 20:
            return {'strength': 0, 'signal': 'Insufficient candle data'}
        
        # Find significant price levels with high volume
        volume_nodes = {}
        for candle in candles:
            high = float(candle[2])
            low = float(candle[3])
            volume = float(candle[5])
            
            # Create price bins
            price_range = high - low
            if price_range > 0:
                mid_price = (high + low) / 2
                # Round to nearest significant level
                price_level = round(mid_price, -int(math.log10(mid_price)) + 2)
                
                if price_level not in volume_nodes:
                    volume_nodes[price_level] = 0
                volume_nodes[price_level] += volume
        
        # Find highest volume nodes (supply/demand zones)
        if not volume_nodes:
            return {'strength': 0, 'signal': 'No volume nodes found'}
        
        sorted_nodes = sorted(volume_nodes.items(), key=lambda x: x[1], reverse=True)
        top_zones = sorted_nodes[:5]  # Top 5 volume zones
        
        signal_strength = 0
        signal_type = ""
        
        # Check proximity to high-volume zones
        for price_level, volume in top_zones:
            distance = abs(current_price - price_level) / current_price * 100
            
            if distance < 1.0:  # Within 1% of major zone
                if direction == 'LONG' and price_level > current_price:
                    signal_strength = 3
                    signal_type = f"Approaching supply zone at ${price_level:.2f}"
                    break
                elif direction == 'SHORT' and price_level < current_price:
                    signal_strength = 3
                    signal_type = f"Approaching demand zone at ${price_level:.2f}"
                    break
            elif distance < 2.0:  # Within 2%
                signal_strength = max(signal_strength, 1)
                signal_type = f"Near volume zone at ${price_level:.2f}"
        
        return {
            'strength': signal_strength,
            'signal': signal_type,
            'top_zones': [(p, v) for p, v in top_zones[:3]]
        }
    
    def _analyze_sentiment_shift(self, current_external, entry_external, direction):
        """Detect significant sentiment shifts since entry"""
        if not current_external or not entry_external:
            return {'strength': 0, 'signal': 'No sentiment data'}
        
        # Calculate sentiment changes
        news_change = current_external.get('news_sentiment', 0) - entry_external.get('news_sentiment', 0)
        fg_change = current_external.get('fear_greed', 50) - entry_external.get('fear_greed', 50)
        social_change = current_external.get('social_sentiment', 0) - entry_external.get('social_sentiment', 0)
        
        signal_strength = 0
        signal_type = ""
        
        # Significant negative sentiment shift for LONG positions
        if direction == 'LONG':
            if news_change < -30 or fg_change < -20:
                signal_strength = 3
                signal_type = "Major negative sentiment shift detected"
            elif news_change < -15 and social_change < -15:
                signal_strength = 2
                signal_type = "Moderate negative sentiment shift"
        
        # Significant positive sentiment shift for SHORT positions  
        else:  # SHORT
            if news_change > 30 or fg_change > 20:
                signal_strength = 3
                signal_type = "Major positive sentiment shift detected"
            elif news_change > 15 and social_change > 15:
                signal_strength = 2
                signal_type = "Moderate positive sentiment shift"
        
        return {
            'strength': signal_strength,
            'signal': signal_type,
            'news_change': news_change,
            'fg_change': fg_change,
            'social_change': social_change
        }
    
    def _analyze_time_based_exit(self, position, profit_pct):
        """Time-based profit taking logic"""
        entry_time = datetime.strptime(position['entry_time'], '%Y-%m-%d %H:%M:%S')
        time_in_trade = datetime.now() - entry_time
        hours_in_trade = time_in_trade.total_seconds() / 3600
        
        signal_strength = 0
        signal_type = ""
        
        # Profit-based time exits
        if profit_pct > 10 and hours_in_trade > 24:  # 10%+ profit after 1 day
            signal_strength = 2
            signal_type = "Excellent profit held for over 24 hours"
        elif profit_pct > 5 and hours_in_trade > 12:  # 5%+ profit after 12 hours
            signal_strength = 1
            signal_type = "Good profit held for extended period"
        elif profit_pct > 15:  # Exceptional profit regardless of time
            signal_strength = 3
            signal_type = "Exceptional profit - consider securing gains"
        
        return {
            'strength': signal_strength,
            'signal': signal_type,
            'hours_in_trade': hours_in_trade,
            'profit_pct': profit_pct
        }
    
    def _determine_profit_action(self, signal_strength, profit_pct, exit_signals):
        """Determine the best profit-taking action"""
        
        # Action thresholds based on signal strength and profit
        if signal_strength >= 8 and profit_pct > 3:
            return {
                'type': 'CLOSE_FULL',
                'reason': f"Strong exit signals ({signal_strength}/20) with {profit_pct:.1f}% profit",
                'details': exit_signals
            }
        
        elif signal_strength >= 6 and profit_pct > 5:
            return {
                'type': 'CLOSE_PARTIAL',
                'reason': f"Moderate signals ({signal_strength}/20) with good profit",
                'partial_exit_pct': 0.5,  # Close 50%
                'details': exit_signals
            }
        
        elif signal_strength >= 4 and profit_pct > 2:
            # Move stop loss to break-even or into profit
            new_stop_percentage = min(profit_pct * 0.5, profit_pct - 1)  # Secure half profit
            return {
                'type': 'MOVE_STOP',
                'reason': f"Weak signals ({signal_strength}/20) - securing profits",
                'new_stop_loss': new_stop_percentage,
                'details': exit_signals
            }
        
        elif profit_pct > 8:
            # Automatic profit protection for large gains
            return {
                'type': 'MOVE_STOP',
                'reason': f"Large profit ({profit_pct:.1f}%) - implementing trailing stop",
                'new_stop_loss': profit_pct * 0.6,  # Secure 60% of profit
                'details': exit_signals
            }
        
        else:
            return {
                'type': 'HOLD',
                'reason': f"Insufficient signals ({signal_strength}/20) or profit ({profit_pct:.1f}%)",
                'details': exit_signals
            }

# Initialize memory systems
trading_memory = TradingMemory()
pattern_learner = PatternLearner()
external_data_manager = ExternalDataManager()
smart_profit_manager = SmartProfitManager()

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
            news_sentiment REAL,
            social_sentiment REAL,
            fear_greed_index INTEGER,
            external_data TEXT,
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
            external_data TEXT,
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
            news_sentiment REAL,
            social_sentiment REAL,
            fear_greed_index INTEGER,
            pattern_hash TEXT,
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
    
    # Memory patterns table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS memory_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_hash TEXT NOT NULL,
            pattern_data TEXT NOT NULL,
            success_rate REAL NOT NULL,
            total_occurrences INTEGER NOT NULL,
            avg_return REAL NOT NULL,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("📁 Enhanced SQLite database with memory systems initialized")

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
        
        if response.status_code == 200:
            balances = response.json()
            
            for balance in balances:
                # Look for USDT balance specifically
                if balance['asset'] in ['USDT', 'USDC', 'BUSD']:
                    available_balance = float(balance['balance'])
                    wallet_balance = float(balance.get('walletBalance', balance['balance']))
                    
                    # Use wallet balance (total) rather than available balance
                    if wallet_balance > 0:
                        return wallet_balance
                    elif available_balance > 0:
                        return available_balance
            
            return portfolio['balance']
        else:
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
            return account_info
        else:
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
    
def get_portfolio_stats():
    """Get comprehensive portfolio statistics from database"""
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    
    # Get initial balance from tracking table
    cursor.execute('SELECT initial_balance FROM portfolio_tracking LIMIT 1')
    initial_result = cursor.fetchone()
    initial_balance = initial_result[0] if initial_result else 1000.0
    
    # Calculate total P&L from all closed trades
    cursor.execute('''
        SELECT 
            COALESCE(SUM(pnl), 0) as total_pnl,
            COUNT(CASE WHEN pnl IS NOT NULL THEN 1 END) as total_trades,
            COUNT(CASE WHEN pnl > 0 THEN 1 END) as winning_trades
        FROM trades 
        WHERE action LIKE "%CLOSE"
    ''')
    
    stats = cursor.fetchone()
    total_pnl = stats[0] if stats[0] is not None else 0.0
    total_closed_trades = stats[1] if stats[1] is not None else 0
    winning_trades = stats[2] if stats[2] is not None else 0
    
    # Calculate win rate
    win_rate = (winning_trades / total_closed_trades * 100) if total_closed_trades > 0 else 0
    
    # Get current balance
    current_balance = get_real_balance() if USE_REAL_TRADING else portfolio['balance']
    
    # Get all trades for history (last 100)
    cursor.execute('''
        SELECT timestamp, position_id, coin, action, direction, price, 
               position_size, leverage, notional_value, stop_loss, take_profit, 
               pnl, pnl_percent, reason, confidence
        FROM trades 
        ORDER BY timestamp DESC 
        LIMIT 100
    ''')
    
    trade_history = []
    for row in cursor.fetchall():
        trade_history.append({
            'time': row[0],
            'position_id': row[1],
            'coin': row[2],
            'action': row[3],
            'direction': row[4],
            'price': row[5],
            'position_size': row[6],
            'leverage': row[7],
            'notional_value': row[8],
            'stop_loss': row[9],
            'take_profit': row[10],
            'pnl': row[11],
            'pnl_percent': row[12],
            'reason': row[13],
            'confidence': row[14]
        })
    
    conn.close()
    
    return {
        'initial_balance': initial_balance,
        'current_balance': current_balance,
        'total_pnl': total_pnl,
        'total_pnl_percent': (total_pnl / initial_balance * 100) if initial_balance > 0 else 0,
        'win_rate': win_rate,
        'total_trades': total_closed_trades,
        'winning_trades': winning_trades,
        'trade_history': trade_history
    }

def get_position_pnl(position, current_price):
    """Calculate current P&L for a position"""
    entry_price = position['entry_price']
    direction = position['direction']
    
    if direction == 'LONG':
        pnl_percent = (current_price - entry_price) / entry_price
    else:
        pnl_percent = (entry_price - current_price) / entry_price
    
    pnl_amount = pnl_percent * position['notional_value']
    
    return {
        'pnl_amount': pnl_amount,
        'pnl_percent': pnl_percent * 100
    }


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
         confidence, reasoning, market_regime, atr_at_entry, external_data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        position.get('atr', 0),
        json.dumps(position.get('external_data', {}))
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
        external_data = json.loads(pos[15]) if len(pos) > 15 and pos[15] else {}
        
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
            'atr': pos[14] if len(pos) > 14 else 0,
            'external_data': external_data
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
        # Get comprehensive portfolio stats
        portfolio_stats = get_portfolio_stats()
        
        # Get current market data
        current_market_data = get_batch_market_data()
        
        # Calculate current position P&L
        position_pnls = {}
        total_unrealized_pnl = 0
        
        for pos_id, position in portfolio['positions'].items():
            coin = position['coin']
            if coin in current_market_data:
                current_price = current_market_data[coin]['price']
                pnl_data = get_position_pnl(position, current_price)
                position_pnls[pos_id] = pnl_data
                total_unrealized_pnl += pnl_data['pnl_amount']
        
        # Enhanced learning insights
        learning_insights = get_learning_insights()
        cost_projections = get_cost_projections()
        
        # Calculate final portfolio value
        final_total_value = portfolio_stats['current_balance'] + total_unrealized_pnl
        final_total_pnl = portfolio_stats['total_pnl'] + total_unrealized_pnl
        
        response_data = {
            # Portfolio basics
            'initial_balance': portfolio_stats['initial_balance'],
            'current_balance': portfolio_stats['current_balance'],
            'total_value': final_total_value,
            'total_pnl': final_total_pnl,
            'total_pnl_percent': (final_total_pnl / portfolio_stats['initial_balance'] * 100),
            'unrealized_pnl': total_unrealized_pnl,
            'realized_pnl': portfolio_stats['total_pnl'],
            
            # Positions with current P&L
            'positions': portfolio['positions'],
            'position_pnls': position_pnls,
            
            # Complete trade history
            'trade_history': portfolio_stats['trade_history'],
            
            # Market data
            'market_data': current_market_data,
            
            # Performance metrics
            'performance_metrics': {
                'win_rate': portfolio_stats['win_rate'],
                'total_trades': portfolio_stats['total_trades'],
                'avg_trade_size': sum(pos['position_size'] for pos in portfolio['positions'].values()) / len(portfolio['positions']) if portfolio['positions'] else 0,
                'active_positions': len(portfolio['positions'])
            },
            
            # Learning and cost data
            'learning_metrics': learning_insights,
            'cost_tracking': cost_projections,
            
            # System info
            'timestamp': datetime.now().isoformat(),
            'trading_mode': 'REAL' if USE_REAL_TRADING else 'PAPER',
            'real_balance': get_real_balance() if USE_REAL_TRADING else None,
            
            # Memory stats
            'memory_stats': {
                'patterns_learned': len(pattern_learner.patterns),
                'trade_memories': len(trading_memory.recent_context),
                'successful_patterns': len([p for p in pattern_learner.patterns.values() 
                                          if p['total_trades'] >= 3 and p['successes'] > p['failures']])
            }
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f"API Status Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
def update_portfolio_balance():
    """Update portfolio balance from real trading or maintain simulated balance"""
    global portfolio
    
    if USE_REAL_TRADING:
        # Sync with real balance
        real_balance = get_real_balance()
        if real_balance != portfolio['balance']:
            print(f"💰 Balance synced: ${portfolio['balance']:.2f} → ${real_balance:.2f}")
            portfolio['balance'] = real_balance
    
    # Update last sync time
    portfolio['last_balance_sync'] = datetime.now().isoformat()

def initialize_portfolio_tracking():
    """Initialize portfolio tracking with proper balance management"""
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    
    # Create initial balance tracking table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolio_tracking (
            id INTEGER PRIMARY KEY,
            initial_balance REAL NOT NULL,
            initialized_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            trading_mode TEXT NOT NULL
        )
    ''')
    
    # Check if we have an initial balance recorded
    cursor.execute('SELECT initial_balance, trading_mode FROM portfolio_tracking LIMIT 1')
    existing = cursor.fetchone()
    
    if not existing:
        # Record initial balance
        if USE_REAL_TRADING:
            initial_balance = get_real_balance()
            trading_mode = 'REAL'
            print(f"🔴 REAL TRADING: Initial balance recorded as ${initial_balance:.2f}")
        else:
            initial_balance = portfolio['balance']
            trading_mode = 'PAPER'
            print(f"📝 PAPER TRADING: Initial balance recorded as ${initial_balance:.2f}")
        
        cursor.execute('''
            INSERT INTO portfolio_tracking (initial_balance, trading_mode)
            VALUES (?, ?)
        ''', (initial_balance, trading_mode))
        
        portfolio['initial_balance'] = initial_balance
    else:
        portfolio['initial_balance'] = existing[0]
        print(f"📊 Loaded existing initial balance: ${existing[0]:.2f} ({existing[1]} mode)")
    
    conn.commit()
    conn.close()

def sync_portfolio_data():
    """Comprehensive portfolio data synchronization"""
    update_portfolio_balance()
    
    # Load any missed positions from database
    db_positions = load_active_positions()
    for pos_id, pos_data in db_positions.items():
        if pos_id not in portfolio['positions']:
            portfolio['positions'][pos_id] = pos_data
            print(f"📊 Loaded missing position: {pos_data['coin']} {pos_data['direction']}")


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

def enhanced_smart_position_analysis(position_id, position, market_data, sentiment, multi_tf_data):
    """Ultra-smart position analysis with advanced profit management"""
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
    
    # 🧠 SMART PROFIT MANAGEMENT - NEW!
    if not should_close and pnl_percent > 0.01:  # If in profit > 1%
        # Get external data for sentiment analysis
        current_external = external_data_manager.get_comprehensive_sentiment(coin)
        
        # Analyze all profit-taking signals
        profit_analysis = smart_profit_manager.analyze_profit_exit_signals(
            position, current_price, market_data[coin], multi_tf_data, current_external
        )
        
        print(f"      🧠 SMART ANALYSIS: {profit_analysis['action']} - {profit_analysis['reason']}")
        
        # Display key signals
        if profit_analysis['signals']:
            top_signals = sorted(profit_analysis['signals'], 
                               key=lambda x: x['strength'], reverse=True)[:3]
            for signal in top_signals:
                if signal['strength'] > 0:
                    print(f"         • {signal['signal']} (Strength: {signal['strength']})")
        
        # Execute smart profit management
        if profit_analysis['action'] == 'CLOSE_FULL':
            should_close = True
            close_reason = f"Smart Exit: {profit_analysis['reason']}"
        
        elif profit_analysis['action'] == 'CLOSE_PARTIAL':
            # Implement partial profit taking
            partial_pct = profit_analysis.get('partial_exit_pct', 0.5)
            partial_amount = position['position_size'] * partial_pct
            partial_pnl = pnl_amount * partial_pct
            
            # Update position size
            position['position_size'] -= partial_amount
            position['notional_value'] -= partial_amount * position['leverage']
            portfolio['balance'] += partial_amount + partial_pnl
            
            print(f"      💰 PARTIAL EXIT: Closed {partial_pct*100:.0f}% for ${partial_pnl:+.2f}")
            print(f"         Remaining position: ${position['position_size']:.2f}")
            
            # Save partial exit to database
            save_trade_to_db({
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'position_id': f"{position_id}_partial",
                'coin': coin,
                'action': f"{direction} PARTIAL_CLOSE",
                'direction': direction,
                'price': current_price,
                'position_size': partial_amount,
                'pnl': partial_pnl,
                'pnl_percent': (partial_pnl / partial_amount) * 100,
                'reason': f"Smart partial exit: {profit_analysis['reason']}"
            })
        
        elif profit_analysis['action'] == 'MOVE_STOP':
            # Implement dynamic stop loss adjustment
            new_stop_pct = profit_analysis.get('new_stop_loss', 0)
            
            if direction == 'LONG':
                new_stop_price = entry_price * (1 + new_stop_pct / 100)
                if new_stop_price > position['stop_loss']:
                    old_stop = position['stop_loss']
                    position['stop_loss'] = new_stop_price
                    print(f"      📍 SMART STOP MOVED: ${old_stop:.2f} → ${new_stop_price:.2f}")
                    print(f"         Securing {new_stop_pct:.1f}% profit")
            else:  # SHORT
                new_stop_price = entry_price * (1 - new_stop_pct / 100)
                if new_stop_price < position['stop_loss']:
                    old_stop = position['stop_loss']
                    position['stop_loss'] = new_stop_price
                    print(f"      📍 SMART STOP MOVED: ${old_stop:.2f} → ${new_stop_price:.2f}")
                    print(f"         Securing {new_stop_pct:.1f}% profit")
    
    # Original trailing stop logic (enhanced)
    elif not should_close and pnl_percent > 0.02:  # If in profit > 2%
        if multi_tf_data and '1h' in multi_tf_data:
            hourly_data = multi_tf_data['1h']
            if 'highs' in hourly_data and 'lows' in hourly_data and 'closes' in hourly_data:
                atr = calculate_atr(hourly_data['highs'], hourly_data['lows'], hourly_data['closes'])
                
                if atr > 0:
                    # Dynamic trailing distance based on volatility
                    if pnl_percent > 0.1:  # 10%+ profit
                        trailing_distance = atr * 1.0  # Tighter trailing
                    elif pnl_percent > 0.05:  # 5%+ profit
                        trailing_distance = atr * 1.3
                    else:
                        trailing_distance = atr * 1.5  # Standard trailing
                    
                    if direction == 'LONG':
                        new_stop = current_price - trailing_distance
                        if new_stop > position['stop_loss'] and new_stop > entry_price:
                            position['stop_loss'] = new_stop
                            print(f"      📍 ATR Trailing stop: ${new_stop:.2f} (ATR: {atr:.2f})")
                    else:  # SHORT
                        new_stop = current_price + trailing_distance
                        if new_stop < position['stop_loss'] and new_stop < entry_price:
                            position['stop_loss'] = new_stop
                            print(f"      📍 ATR Trailing stop: ${new_stop:.2f} (ATR: {atr:.2f})")
    
    # Final close decision
    if should_close:
        close_position(position_id, position, current_price, close_reason, pnl_amount)
    
    return pnl_amount

def ai_trade_decision_with_memory(coin, market_data, multi_tf_data, learning_insights):
    """Enhanced AI trading decision with memory and external data"""
    if not client:
        return None
    
    current_price = market_data[coin]['price']
    tf_alignment = analyze_timeframe_alignment(multi_tf_data)
    
    # Skip if timeframes not aligned
    if not tf_alignment['aligned']:
        return {'direction': 'SKIP', 'reason': 'Timeframes not aligned'}
    
    # Get external sentiment data
    external_data = external_data_manager.get_comprehensive_sentiment(coin)
    
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
    
    # RSI for pattern matching
    rsi = calculate_rsi(hourly_data.get('closes', [])) if hourly_data.get('closes') else 50
    
    # Market conditions for pattern learning
    market_conditions = {
        'regime': market_regime['regime'],
        'tf_alignment': tf_alignment['direction'],
        'volume_trend': 'increasing' if market_data[coin]['volume'] > 1000000 else 'normal',
        'market_data': market_data[coin]
    }
    
    # Technical indicators for pattern matching
    indicators = {
        'rsi': rsi,
        'atr': atr,
        'volume_ratio': market_data[coin]['volume'] / market_data[coin].get('quote_volume', 1),
        'bb_width': market_regime['indicators'].get('bb_width', 0),
        'support_levels': sr_levels.get('support', []),
        'resistance_levels': sr_levels.get('resistance', [])
    }
    
    # Get pattern-based recommendations
    pattern_recommendations = pattern_learner.get_pattern_recommendation(
        market_conditions, indicators, external_data, coin
    )
    
    # Dynamic stops calculation
    dynamic_stops = calculate_dynamic_stops(
        current_price, 
        atr, 
        sr_levels.get('support', []), 
        sr_levels.get('resistance', []),
        'LONG' if tf_alignment['direction'] in ['bullish', 'strong_bullish'] else 'SHORT',
        market_regime['regime']
    )
    
    # Enhanced analysis with memory and external data
    analysis_prompt = f"""
{coin} ADVANCED TRADING ANALYSIS WITH MEMORY & EXTERNAL DATA

CURRENT MARKET:
- Price: ${current_price:.2f} ({market_data[coin]['change_24h']:+.1f}% 24h)
- Volume: {market_data[coin]['volume']/1000000:.1f}M
- Market Regime: {market_regime['regime']} (confidence: {market_regime['confidence']:.0f}%)
- Timeframe Alignment: {tf_alignment['direction']} (score: {tf_alignment['score']:.1f})

TECHNICAL INDICATORS:
- RSI: {rsi:.1f}
- ATR: ${atr:.2f}
- VWAP: ${vwap:.2f}
- Support: {', '.join([f'${s:.2f}' for s in sr_levels['support'][:2]])}
- Resistance: {', '.join([f'${r:.2f}' for r in sr_levels['resistance'][:2]])}

EXTERNAL SENTIMENT:
- News Sentiment: {external_data['news_sentiment']}/100
- Social Sentiment: {external_data['social_sentiment']}/100
- Fear & Greed Index: {external_data['fear_greed']}/100

PATTERN ANALYSIS:
{json.dumps(pattern_recommendations, indent=2) if pattern_recommendations else "No similar patterns found"}

RISK MANAGEMENT:
- Suggested SL: {dynamic_stops['sl_percentage']:.1f}%
- Suggested TP: {dynamic_stops['tp_percentage']:.1f}%
- Risk/Reward: {dynamic_stops['risk_reward']:.1f}

LEARNING INSIGHTS:
- Historical Win Rate: {learning_insights.get('win_rate', 0):.0f}%
- Best Leverage: {learning_insights.get('best_leverage', 10)}x

Consider all factors: technical analysis, external sentiment, learned patterns, and risk management.
Provide decision with high confidence only when multiple factors align.
"""

    # Use memory-enhanced AI decision
    decision_response = trading_memory.get_decision_with_memory(coin, analysis_prompt)
    
    if not decision_response:
        return None
    
    # Parse the enhanced response
    trade_params = {}
    
    # Extract decision from response
    decision_lines = decision_response.split('\n')
    for line in decision_lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().upper()
            value = value.strip()
            
            if 'DECISION' in key or 'ACTION' in key:
                if 'LONG' in value.upper():
                    trade_params['direction'] = 'LONG'
                elif 'SHORT' in value.upper():
                    trade_params['direction'] = 'SHORT'
                else:
                    trade_params['direction'] = 'SKIP'
            elif 'LEVERAGE' in key:
                leverage = int(''.join(filter(str.isdigit, value)))
                trade_params['leverage'] = min(20, max(5, leverage))
            elif 'SIZE' in key or 'POSITION' in key:
                size_match = ''.join(filter(lambda x: x.isdigit() or x == '.', value))
                if size_match:
                    size = float(size_match)
                    trade_params['position_size'] = min(0.10, max(0.03, size / 100))
            elif 'CONFIDENCE' in key:
                conf_match = ''.join(filter(str.isdigit, value))
                if conf_match:
                    trade_params['confidence'] = min(10, max(1, int(conf_match)))
            elif 'REASON' in key:
                trade_params['reasoning'] = value[:200]
    
    # Set defaults if not parsed
    if 'direction' not in trade_params:
        trade_params['direction'] = 'SKIP'
    if 'leverage' not in trade_params:
        trade_params['leverage'] = 10
    if 'position_size' not in trade_params:
        trade_params['position_size'] = 0.05
    if 'confidence' not in trade_params:
        trade_params['confidence'] = 5
    if 'reasoning' not in trade_params:
        trade_params['reasoning'] = f"Memory-based decision: {market_regime['regime']} market"
    
    # Adjust leverage based on confidence and pattern recommendations
    if pattern_recommendations:
        best_pattern = max(pattern_recommendations.values(), 
                          key=lambda x: x.get('confidence', 0))
        if best_pattern.get('confidence', 0) > 0.8:
            trade_params['confidence'] = min(10, trade_params['confidence'] + 2)
    
    # Confidence-based leverage override
    if trade_params['confidence'] >= 9:
        trade_params['leverage'] = 20
    elif trade_params['confidence'] >= 8:
        trade_params['leverage'] = 15
    elif trade_params['confidence'] >= 7:
        trade_params['leverage'] = 12
    else:
        trade_params['leverage'] = max(5, trade_params['leverage'])
    
    # Add calculated data
    trade_params['stop_loss'] = dynamic_stops['stop_loss']
    trade_params['take_profit'] = dynamic_stops['take_profit']
    trade_params['sl_percentage'] = dynamic_stops['sl_percentage']
    trade_params['tp_percentage'] = dynamic_stops['tp_percentage']
    trade_params['market_regime'] = market_regime['regime']
    trade_params['atr'] = atr
    trade_params['external_data'] = external_data
    trade_params['pattern_recommendations'] = pattern_recommendations
    
    return trade_params

def execute_trade_with_memory(coin, trade_params, current_price):
    """Execute trade with memory tracking"""
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
    
    # Position sizing
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
    
    # Use dynamic stops
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
    
    # Store position data with external data
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
        'quantity': quantity,
        'external_data': trade_params.get('external_data', {}),
        'pattern_recommendations': trade_params.get('pattern_recommendations', {})
    }
    
    portfolio['positions'][position_id] = position
    portfolio['balance'] -= position_value
    
    # Save to database with external data
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
        'reason': trade_params['reasoning'],
        'news_sentiment': trade_params.get('external_data', {}).get('news_sentiment', 0),
        'social_sentiment': trade_params.get('external_data', {}).get('social_sentiment', 0),
        'fear_greed_index': trade_params.get('external_data', {}).get('fear_greed', 50),
        'external_data': json.dumps(trade_params.get('external_data', {}))
    })
    
    # Display enhanced trade info
    emoji = "📈" if trade_params['direction'] == 'LONG' else "📉"
    risk_reward = trade_params.get('tp_percentage', 0) / trade_params.get('sl_percentage', 1) if trade_params.get('sl_percentage', 0) > 0 else 0
    
    trading_mode = "🔴 REAL" if USE_REAL_TRADING else "📝 PAPER"
    
    print(f"\n   {emoji} {trade_params['direction']} {coin}: ${position_value:.2f} @ {leverage}x {trading_mode}")
    print(f"      Market: {trade_params.get('market_regime', 'UNKNOWN')} | Confidence: {trade_params['confidence']}/10")
    print(f"      SL: ${stop_loss_price:.2f} (-{trade_params.get('sl_percentage', 0):.1f}%) | TP: ${take_profit_price:.2f} (+{trade_params.get('tp_percentage', 0):.1f}%)")
    print(f"      Risk/Reward: 1:{risk_reward:.1f}")
    print(f"      External: News:{trade_params.get('external_data', {}).get('news_sentiment', 0):+.0f} | F&G:{trade_params.get('external_data', {}).get('fear_greed', 50)}")
    print(f"      Reason: {trade_params['reasoning'][:60]}...")

def close_position_with_memory(position_id, position, current_price, reason, pnl_amount):
    """Close position and record outcome for learning"""
    global portfolio
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    pnl_percent = (pnl_amount / position['position_size']) * 100
    was_successful = pnl_amount > 0
    
    # REAL TRADING: Close the position
    if USE_REAL_TRADING:
        symbol = f"{position['coin']}USDT"
        side = 'SELL' if position['direction'] == 'LONG' else 'BUY'
        quantity = position.get('quantity', position['notional_value'] / current_price)
        
        close_result = close_real_position(symbol, side, quantity)
        
        if close_result:
            print(f"✅ REAL POSITION CLOSED: {close_result.get('orderId')}")
        else:
            print(f"❌ REAL CLOSE FAILED for {position['coin']}")
    
    # Record outcome in memory systems
    coin = position['coin']
    
    # Add outcome to trading memory
    outcome_details = f"P&L: ${pnl_amount:+.2f} ({pnl_percent:+.1f}%) | Reason: {reason} | {'✅ Profitable' if was_successful else '❌ Loss'}"
    trading_memory.add_trade_outcome(coin, outcome_details)
    
    # Record pattern for learning
    if 'external_data' in position and 'pattern_recommendations' in position:
        market_conditions = {
            'regime': position.get('market_regime', 'UNKNOWN'),
            'market_data': {'price': current_price}
        }
        
        indicators = {
            'rsi': 50,  # Would need to calculate actual RSI
            'atr': position.get('atr', 0)
        }
        
        external_data = position.get('external_data', {})
        
        pattern_learner.record_pattern_outcome(
            market_conditions,
            indicators, 
            external_data,
            position['direction'],
            was_successful,
            pnl_percent,
            coin
        )
    
    # Save to learning patterns table
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO learning_patterns 
        (coin, direction, leverage, confidence, duration_target, profitable, 
         pnl_percent, market_conditions, news_sentiment, social_sentiment, 
         fear_greed_index, pattern_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        coin,
        position['direction'],
        position['leverage'],
        position['confidence'],
        position.get('duration_target', 'SWING'),
        was_successful,
        pnl_percent,
        position.get('market_regime', 'UNKNOWN'),
        position.get('external_data', {}).get('news_sentiment', 0),
        position.get('external_data', {}).get('social_sentiment', 0),
        position.get('external_data', {}).get('fear_greed', 50),
        f"{coin}_{position['direction']}_{position.get('market_regime', 'UNKNOWN')}"
    ))
    
    conn.commit()
    conn.close()
    
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
        'duration': str(datetime.now() - datetime.strptime(position['entry_time'], '%Y-%m-%d %H:%M:%S')),
        'news_sentiment': position.get('external_data', {}).get('news_sentiment', 0),
        'social_sentiment': position.get('external_data', {}).get('social_sentiment', 0),
        'fear_greed_index': position.get('external_data', {}).get('fear_greed', 50)
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
    trading_mode = "🔴 REAL" if USE_REAL_TRADING else "📝 PAPER"
    
    print(f"   {emoji} CLOSED {position['coin']}: ${pnl_amount:+.2f} ({pnl_percent:+.1f}%) - {reason} {trading_mode}")
    print(f"      🧠 Pattern learned and stored in memory for future decisions")

# Update function references
def close_position(position_id, position, current_price, reason, pnl_amount):
    """Wrapper for memory-enhanced position closing"""
    return close_position_with_memory(position_id, position, current_price, reason, pnl_amount)

def ai_trade_decision(coin, market_data, multi_tf_data, learning_insights):
    """Wrapper for memory-enhanced trade decisions"""
    return ai_trade_decision_with_memory(coin, market_data, multi_tf_data, learning_insights)

def execute_trade(coin, trade_params, current_price):
    """Wrapper for memory-enhanced trade execution"""
    return execute_trade_with_memory(coin, trade_params, current_price)

def save_trade_to_db(trade_record):
    """Enhanced trade saving with external data"""
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT OR REPLACE INTO trades 
        (timestamp, position_id, coin, action, direction, price, position_size, 
         leverage, notional_value, stop_loss, take_profit, pnl, pnl_percent, 
         duration, reason, confidence, profitable, news_sentiment, social_sentiment,
         fear_greed_index, external_data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        trade_record.get('pnl', 0) > 0 if 'pnl' in trade_record else None,
        trade_record.get('news_sentiment', 0),
        trade_record.get('social_sentiment', 0),
        trade_record.get('fear_greed_index', 50),
        trade_record.get('external_data', '{}')
    ))
    
    conn.commit()
    conn.close()

def get_learning_insights():
    """Enhanced learning insights with memory data"""
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
    
    # External data correlations
    cursor.execute('''
        SELECT 
            CASE 
                WHEN news_sentiment > 20 THEN 'positive_news'
                WHEN news_sentiment < -20 THEN 'negative_news'
                ELSE 'neutral_news'
            END as news_category,
            AVG(pnl_percent) as avg_pnl,
            COUNT(*) as count
        FROM trades
        WHERE pnl IS NOT NULL
        GROUP BY news_category
        HAVING count >= 3
    ''')
    
    news_performance = cursor.fetchall()
    insights['news_correlation'] = {row[0]: {'avg_pnl': row[1], 'count': row[2]} for row in news_performance}
    
    # Fear & Greed correlations
    cursor.execute('''
        SELECT 
            CASE 
                WHEN fear_greed_index < 25 THEN 'extreme_fear'
                WHEN fear_greed_index < 45 THEN 'fear'
                WHEN fear_greed_index < 55 THEN 'neutral'
                WHEN fear_greed_index < 75 THEN 'greed'
                ELSE 'extreme_greed'
            END as fg_category,
            AVG(pnl_percent) as avg_pnl,
            COUNT(*) as count
        FROM trades
        WHERE pnl IS NOT NULL
        GROUP BY fg_category
        HAVING count >= 3
    ''')
    
    fg_performance = cursor.fetchall()
    insights['fear_greed_correlation'] = {row[0]: {'avg_pnl': row[1], 'count': row[2]} for row in fg_performance}
    
    # Memory system stats
    insights['memory_stats'] = {
        'patterns_learned': len(pattern_learner.patterns),
        'conversation_length': len(trading_memory.conversation_history),
        'successful_patterns': len([p for p in pattern_learner.patterns.values() 
                                  if p['total_trades'] >= 3 and p['successes'] > p['failures']])
    }
    
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
    """Enhanced auto-tuning with memory and external data"""
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    
    # Get last N trades with external data
    cursor.execute('''
        SELECT stop_loss, take_profit, pnl_percent, confidence, news_sentiment, 
               fear_greed_index, social_sentiment
        FROM trades
        WHERE stop_loss IS NOT NULL AND take_profit IS NOT NULL AND pnl_percent IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (window,))
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return
    
    # Enhanced tuning with external factors
    best_score = -9999
    best_params = {}
    
    for sl_mult in [1.0, 1.2, 1.5, 1.8, 2.0]:
        for tp_mult in [1.5, 2.0, 2.5, 3.0, 4.0]:
            for conf in [5, 6, 7, 8]:
                for news_threshold in [-20, 0, 20]:  # News sentiment threshold
                    # Filter trades by parameters
                    filtered = [row for row in rows 
                              if row[3] >= conf and row[4] >= news_threshold]
                    
                    if len(filtered) < 10:
                        continue
                    
                    profits = [row[2] for row in filtered]
                    if len(profits) < 2:
                        continue
                    
                    mean = sum(profits) / len(profits)
                    std = statistics.stdev(profits)
                    score = mean / std if std > 0 else mean
                    
                    # Bonus for external data correlation
                    positive_news_trades = [row for row in filtered if row[4] > 10]
                    if positive_news_trades:
                        news_avg = sum(row[2] for row in positive_news_trades) / len(positive_news_trades)
                        if news_avg > 0:
                            score += 0.1  # Bonus for positive news correlation
                    
                    # Penalty for too few trades
                    score -= 0.05 * (20 - len(filtered)) if len(filtered) < 20 else 0
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'sl_mult': sl_mult, 
                            'tp_mult': tp_mult, 
                            'conf': conf,
                            'news_threshold': news_threshold
                        }
    
    # Save best parameters
    if best_params:
        set_hyperparameter('sl_multiplier', best_params['sl_mult'])
        set_hyperparameter('tp_multiplier', best_params['tp_mult'])
        set_hyperparameter('min_confidence', best_params['conf'])
        set_hyperparameter('news_threshold', best_params['news_threshold'])
        
        print(f"\n🔧 Enhanced Auto-tuning:")
        print(f"   SL: {best_params['sl_mult']}x | TP: {best_params['tp_mult']}x")
        print(f"   Min Confidence: {best_params['conf']}/10")
        print(f"   News Threshold: {best_params['news_threshold']}")
        print(f"   Sharpe Score: {best_score:.2f}")

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
    """Main bot loop with memory and external data"""
    # Initialize
    init_database()
    initialize_portfolio_tracking()  # ADD THIS LINE
    portfolio['positions'] = load_active_positions()
    sync_portfolio_data()
    
    print("🚀 SUPER SMART TRADING BOT - MEMORY & EXTERNAL DATA MODE")
    print("🧠 Memory Systems:")
    print("   • Trading Memory: Conversation-based learning")
    print("   • Pattern Learner: Similarity matching & outcomes")
    print("   • External Data: News, Fear/Greed, Social sentiment")
    print("⚡ Quick checks: Every 15 seconds (positions + trailing stops)")
    print("🧠 Full analysis: Every 2 minutes (complete scan + memory)")
    print("📊 Advanced Features:")
    print("   • Volume Profile & Liquidity Analysis")
    print("   • Support/Resistance Detection")
    print("   • Market Regime Detection (Trending/Ranging/Volatile)")
    print("   • Dynamic Stop Loss & Take Profit")
    print("   • Trailing Stops for Winners")
    print("   • Multi-Indicator Confluence")
    print("   • Real News Sentiment Analysis")
    print("   • Fear & Greed Index Integration")
    print("   • Pattern Memory & Recognition")
    print("💰 Max positions: 4 concurrent")
    print("🎯 Min confidence: 6/10 (auto-tuned)")
    print("📊 Coins: BTC, ETH, SOL, BNB, ADA, DOGE, AVAX, TAO, LINK, DOT, UNI, FET")
    print("🌐 Dashboard: http://localhost:5000")
    print("="*80)
    
    # Price histories for technical analysis
    target_coins = ['BTC', 'ETH', 'SOL', 'BNB', 'ADA', 'DOGE', 
                   'AVAX', 'TAO', 'LINK', 'DOT', 'UNI', 'FET']
    price_histories = {coin: [] for coin in target_coins}
    
    # Memory and pattern tracking
    last_external_data_update = 0
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
                sync_portfolio_data()
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
                    memory_info = f" | 🧠 {len(pattern_learner.patterns)} patterns"
                    status = f"⚡ Quick #{quick_checks_count}: {len(portfolio['positions'])} positions{memory_info}"
                    if positions_closed > 0:
                        status += f" | {positions_closed} closed"
                    print(f"\r{status}", end="", flush=True)
            
            # ========== FULL ANALYSIS MODE (Every 2 minutes) ==========
            if current_time - last_full_analysis >= FULL_ANALYSIS_INTERVAL:
                full_analyses_count += 1
                print(f"\n\n{'='*80}")
                print(f"🧠 SUPER SMART ANALYSIS #{full_analyses_count}")
                print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                mode_indicator = "🔴 REAL TRADING" if USE_REAL_TRADING else "📝 PAPER TRADING"
                print(f"Mode: {mode_indicator}")
                print(f"Current Balance: ${portfolio['balance']:.2f}")
                print(f"Memory: {len(trading_memory.conversation_history)} conversations | {len(pattern_learner.patterns)} patterns learned")
                print("="*80)
                
                quick_checks_count = 0  # Reset counter
                
                # Get comprehensive market data
                market_data = get_batch_market_data()
                
                # Update price histories
                for coin in target_coins:
                    if coin in market_data:
                        price_histories[coin].append(market_data[coin]['price'])
                        if len(price_histories[coin]) > 100:
                            price_histories[coin] = price_histories[coin][-100:]
                
                # Update external data every 10 minutes
                if current_time - last_external_data_update >= 600:
                    print("📡 Updating external sentiment data...")
                    last_external_data_update = current_time
                
                # Enhanced market overview with sentiment
                print("\n📊 Market Overview with External Sentiment:")
                for coin in target_coins[:6]:
                    if coin in market_data:
                        data = market_data[coin]
                        
                        # Get external sentiment
                        external_data = external_data_manager.get_comprehensive_sentiment(coin)
                        
                        # Get multi-timeframe data
                        symbol = f"{coin}USDT"
                        coin_tf_data = get_multi_timeframe_data(symbol)
                        
                        # Calculate indicators
                        rsi = calculate_rsi(price_histories[coin]) if len(price_histories[coin]) > 14 else 50
                        
                        # Detect market regime
                        regime_data = detect_market_regime(coin_tf_data, data['price'])
                        regime = regime_data['regime']
                        
                        # Trend emoji
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
                        
                        # Sentiment indicators
                        news_emoji = "📰✅" if external_data['news_sentiment'] > 10 else "📰❌" if external_data['news_sentiment'] < -10 else "📰➖"
                        fg_emoji = "😱" if external_data['fear_greed'] < 25 else "😰" if external_data['fear_greed'] < 45 else "😐" if external_data['fear_greed'] < 55 else "🤑" if external_data['fear_greed'] < 75 else "🚀"
                        
                        print(f"   {coin}: ${data['price']:.2f} {trend} ({data['change_24h']:+.1f}%) RSI:{rsi:.0f}")
                        print(f"      {news_emoji} News:{external_data['news_sentiment']:+.0f} | {fg_emoji} F&G:{external_data['fear_greed']}")
                
                # Advanced position management with memory
                if portfolio['positions']:
                    print(f"\n📈 Advanced Position Management with Memory:")
                    sentiment_scores = {}
                    
                    # Calculate enhanced sentiment for all coins
                    for coin in market_data:
                        rsi = calculate_rsi(price_histories[coin]) if len(price_histories[coin]) > 14 else 50
                        momentum = market_data[coin]['change_24h']
                        
                        # Get external data
                        external_data = external_data_manager.get_comprehensive_sentiment(coin)
                        
                        # Complex sentiment calculation with external data
                        sentiment = 0
                        
                        # Technical sentiment
                        if rsi < 30:
                            sentiment += 30
                        elif rsi > 70:
                            sentiment -= 30
                        else:
                            sentiment += (50 - rsi) * 0.4
                        
                        # Momentum
                        sentiment += min(max(momentum * 1.5, -25), 25)
                        
                        # External sentiment (weighted)
                        sentiment += external_data['news_sentiment'] * 0.3
                        sentiment += (external_data['fear_greed'] - 50) * 0.2
                        sentiment += external_data['social_sentiment'] * 0.2
                        
                        sentiment_scores[coin] = sentiment
                    
                    # Analyze each position with memory
                    for pos_id, position in list(portfolio['positions'].items()):
                        coin = position['coin']
                        if coin in market_data:
                            # Get fresh data
                            symbol = f"{coin}USDT"
                            position_tf_data = get_multi_timeframe_data(symbol)
                            
                            current_price = market_data[coin]['price']
                            entry_price = position['entry_price']
                            
                            # Calculate P&L
                            if position['direction'] == 'LONG':
                                pnl_pct = (current_price - entry_price) / entry_price
                            else:
                                pnl_pct = (entry_price - current_price) / entry_price
                            
                            pnl_amount = pnl_pct * position['notional_value']
                            
                            # Calculate distances
                            sl_distance = abs(current_price - position['stop_loss']) / current_price * 100
                            tp_distance = abs(position['take_profit'] - current_price) / current_price * 100
                            
                            # Enhanced display with memory insights
                            entry_external = position.get('external_data', {})
                            current_external = external_data_manager.get_comprehensive_sentiment(coin)
                            
                            print(f"   • {position['direction']} {coin}: P&L ${pnl_amount:+.2f} ({pnl_pct*100:+.1f}%)")
                            print(f"     SL: -{sl_distance:.1f}% | TP: +{tp_distance:.1f}% | Sentiment: {sentiment_scores.get(coin, 0):.0f}")
                            print(f"     Entry News: {entry_external.get('news_sentiment', 0):+.0f} → Current: {current_external['news_sentiment']:+.0f}")
                            
                            # Memory-enhanced position analysis
                            enhanced_smart_position_analysis(pos_id, position, market_data, sentiment_scores.get(coin, 0), position_tf_data)
                
                # Look for new opportunities with memory
                if len(portfolio['positions']) < MAX_CONCURRENT_POSITIONS:
                    if all(len(history) >= 20 for history in price_histories.values()):
                        learning_insights = get_learning_insights()
                        
                        print(f"\n🤖 Super Smart Opportunity Scan with Memory:")
                        print(f"   Historical Win Rate: {learning_insights['win_rate']:.1f}%")
                        print(f"   Patterns Learned: {learning_insights['memory_stats']['patterns_learned']}")
                        print(f"   Successful Patterns: {learning_insights['memory_stats']['successful_patterns']}")
                        
                        # Show external data correlations
                        if 'news_correlation' in learning_insights:
                            best_news = max(learning_insights['news_correlation'].items(), 
                                          key=lambda x: x[1]['avg_pnl'], default=None)
                            if best_news:
                                print(f"   Best News Condition: {best_news[0]} ({best_news[1]['avg_pnl']:+.1f}% avg)")
                        
                        if 'fear_greed_correlation' in learning_insights:
                            best_fg = max(learning_insights['fear_greed_correlation'].items(), 
                                        key=lambda x: x[1]['avg_pnl'], default=None)
                            if best_fg:
                                print(f"   Best F&G Condition: {best_fg[0]} ({best_fg[1]['avg_pnl']:+.1f}% avg)")
                        
                        print(f"   Scanning {len(target_coins)} coins with memory & external data...")
                        
                        opportunities_found = 0
                        opportunities_analyzed = 0
                        
                        for coin in target_coins:
                            # Skip existing positions
                            if any(p['coin'] == coin for p in portfolio['positions'].values()):
                                continue
                            
                            opportunities_analyzed += 1
                            
                            # Get comprehensive data
                            symbol = f"{coin}USDT"
                            multi_tf_data = get_multi_timeframe_data(symbol)
                            
                            # Skip if regime is unclear
                            regime_data = detect_market_regime(multi_tf_data, market_data[coin]['price'])
                            if regime_data['regime'] == 'TRANSITIONAL' and regime_data['confidence'] < 60:
                                continue
                            
                            # Memory-enhanced AI decision
                            trade_params = ai_trade_decision(coin, market_data, multi_tf_data, learning_insights)
                            
                            if trade_params and trade_params['direction'] != 'SKIP':
                                if trade_params['confidence'] >= get_hyperparameter('min_confidence', 6):
                                    # Check external sentiment threshold
                                    external_data = trade_params.get('external_data', {})
                                    news_threshold = get_hyperparameter('news_threshold', 0)
                                    
                                    if external_data.get('news_sentiment', 0) >= news_threshold:
                                        risk_reward = trade_params.get('tp_percentage', 0) / trade_params.get('sl_percentage', 1)
                                        
                                        if risk_reward >= 1.5:
                                            print(f"   ✅ Smart Opportunity: {trade_params['direction']} {coin}")
                                            print(f"      Confidence: {trade_params['confidence']}/10 | R:R: 1:{risk_reward:.1f}")
                                            print(f"      News: {external_data.get('news_sentiment', 0):+.0f} | F&G: {external_data.get('fear_greed', 50)}")
                                            
                                            # Show pattern recommendation if available
                                            if trade_params.get('pattern_recommendations'):
                                                best_pattern = max(trade_params['pattern_recommendations'].values(), 
                                                                 key=lambda x: x.get('confidence', 0))
                                                print(f"      Memory: {best_pattern.get('reason', 'New pattern')}")
                                            
                                            execute_trade(coin, trade_params, market_data[coin]['price'])
                                            opportunities_found += 1
                                            
                                            if opportunities_found >= 2:
                                                print(f"   📊 Trade limit reached (2 per cycle)")
                                                break
                                        else:
                                            print(f"   ⏭️ {coin}: Poor R:R (1:{risk_reward:.1f})")
                                    else:
                                        print(f"   ⏭️ {coin}: News sentiment too low ({external_data.get('news_sentiment', 0)})")
                                else:
                                    print(f"   ⏭️ {coin}: Low confidence ({trade_params['confidence']}/10)")
                        
                        if opportunities_found == 0:
                            print(f"   No high-confidence opportunities from {opportunities_analyzed} coins analyzed")
                    else:
                        data_ready = min(len(h) for h in price_histories.values())
                        print(f"\n📈 Building price history... ({data_ready}/20 candles)")
                else:
                    print(f"\n⚠️ Position limit reached ({len(portfolio['positions'])}/{MAX_CONCURRENT_POSITIONS})")
                
                # Enhanced portfolio performance
                total_value = calculate_portfolio_value(market_data)
                pnl = total_value - 1000
                pnl_pct = (pnl / 1000) * 100
                
                # Enhanced Sharpe with external factors
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
                
                print(f"\n💼 ENHANCED PORTFOLIO PERFORMANCE:")
                print(f"   Total Value: ${total_value:.2f}")
                print(f"   P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)")
                print(f"   Free Balance: ${portfolio['balance']:.2f}")
                print(f"   Active Positions: {len(portfolio['positions'])}")
                print(f"   Sharpe Ratio: {sharpe:.2f}")
                print(f"   🧠 Memory Intelligence: {len(pattern_learner.patterns)} patterns | {len(trading_memory.recent_context)} decisions")
                
                # Cost tracking
                costs = get_cost_projections()
                print(f"\n💰 COST ANALYSIS:")
                print(f"   Session: ${costs['current']['total']:.4f} ({costs['current']['api_calls']} API calls)")
                print(f"   Weekly Projection: ${costs['projections']['weekly']['total']:.2f}")
                print(f"   Monthly Projection: ${costs['projections']['monthly']['total']:.2f}")
                print(f"   Net Profit (after costs): ${pnl - costs['current']['total']:.2f}")

                # Enhanced auto-tuning every 5 cycles
                if full_analyses_count % 5 == 0:
                    auto_tune_hyperparameters(window=50)
                
                last_full_analysis = current_time
                print("="*80)
            
            # Sleep for quick check interval
            time.sleep(QUICK_CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            print("\n\n🛑 Super Smart Bot stopped by user")
            print(f"🧠 Final Memory Stats:")
            print(f"   Patterns Learned: {len(pattern_learner.patterns)}")
            print(f"   Successful Patterns: {len([p for p in pattern_learner.patterns.values() if p['successes'] > p['failures']])}")
            print(f"   Conversation History: {len(trading_memory.conversation_history)} messages")
            print(f"   Trade Memories: {len(trading_memory.recent_context)} decisions")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(QUICK_CHECK_INTERVAL)

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 SUPER SMART TRADING BOT INITIALIZATION")
    print("="*80)
    
    # Check API keys
    missing_keys = []
    if not NEWS_API_KEY:
        missing_keys.append("NEWS_API_KEY (get from newsapi.org)")
    if not ALPHA_VANTAGE_KEY:
        missing_keys.append("ALPHA_VANTAGE_KEY (get from alphavantage.co)")
    
    if missing_keys:
        print("⚠️  OPTIONAL API KEYS MISSING:")
        for key in missing_keys:
            print(f"   • {key}")
        print("   Bot will work with limited external data")
        print("   Add these keys for full sentiment analysis")
        print()
    
    # Initialize memory systems
    print("🧠 Initializing Memory Systems...")
    print(f"   ✅ Trading Memory: Conversation-based learning")
    print(f"   ✅ Pattern Learner: Similarity matching with {len(pattern_learner.embeddings)} embeddings")
    print(f"   ✅ External Data Manager: News, F&G, Social sentiment")
    print()
    
    # Test external data connection
    print("📡 Testing External Data Sources...")
    try:
        fear_greed = external_data_manager.get_fear_greed_index()
        print(f"   ✅ Fear & Greed Index: {fear_greed}/100")
    except:
        print(f"   ⚠️  Fear & Greed Index: Connection failed")
    
    if NEWS_API_KEY:
        try:
            btc_sentiment = external_data_manager.get_news_sentiment('BTC')
            print(f"   ✅ News Sentiment (BTC): {btc_sentiment}/100")
        except:
            print(f"   ⚠️  News API: Connection failed")
    else:
        print(f"   ⏭️  News API: Skipped (no API key)")
    
    print()
    
    # Start Flask dashboard in background
    print("🌐 Starting Web Dashboard...")
    flask_thread = threading.Thread(target=run_flask_app, daemon=True)
    flask_thread.start()
    print("   ✅ Dashboard running at http://localhost:5000")
    print()
    
    # Display final configuration
    print("⚙️  FINAL CONFIGURATION:")
    print(f"   Trading Mode: {'🔴 REAL' if USE_REAL_TRADING else '📝 PAPER'}")
    print(f"   Max Positions: {MAX_CONCURRENT_POSITIONS}")
    print(f"   Min Confidence: {MIN_CONFIDENCE_THRESHOLD}/10")
    print(f"   Check Interval: {QUICK_CHECK_INTERVAL}s quick, {FULL_ANALYSIS_INTERVAL}s full")
    print(f"   Memory Enabled: ✅")
    print(f"   External Data: ✅")
    print(f"   Auto-tuning: ✅")
    print()
    
    input("Press ENTER to start the Super Smart Trading Bot...")
    
    # Start the enhanced bot
    run_enhanced_bot()