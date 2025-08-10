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
import numpy as np


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
MAX_CONCURRENT_POSITIONS = 6  # Increased from 4 to 6
MIN_CONFIDENCE_THRESHOLD = 6
BATCH_API_WEIGHT_LIMIT = 100

# Enhanced balance management for multiple positions
POSITION_SCALING = {
    0: 0.15,  # 1st position: 15% of balance
    1: 0.12,  # 2nd position: 12% of balance  
    2: 0.10,  # 3rd position: 10% of balance
    3: 0.08,  # 4th position: 8% of balance
    4: 0.06,  # 5th position: 6% of balance
    5: 0.05   # 6th position: 5% of balance
}

# Enhanced weighting system for factors
FACTOR_WEIGHTS = {
    'timeframe_alignment': 0.25,    # Most important
    'market_regime': 0.20,          # Very important
    'momentum': 0.15,               # Important
    'volume_profile': 0.12,         # Important
    'technical_indicators': 0.10,   # Moderate
    'volatility': 0.08,             # Moderate
    'liquidity': 0.06,              # Less critical
    'divergences': 0.04             # Least critical but valuable
}

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

# ========== NEW FEATURE 1: ORDER BOOK ANALYSIS ==========
def get_order_book_analysis(symbol):
    """Analyze order book for liquidity and support/resistance"""
    try:
        depth_url = f"https://api.binance.com/api/v3/depth"
        response = requests.get(depth_url, params={'symbol': symbol, 'limit': 100})
        
        if response.status_code != 200:
            return {'liquidity_score': 50, 'imbalance_ratio': 1.0, 'spread': 0}
        
        data = response.json()
        bids = [[float(price), float(qty)] for price, qty in data['bids'][:20]]
        asks = [[float(price), float(qty)] for price, qty in data['asks'][:20]]
        
        if not bids or not asks:
            return {'liquidity_score': 50, 'imbalance_ratio': 1.0, 'spread': 0}
        
        # Calculate spread
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        spread_pct = ((best_ask - best_bid) / best_bid) * 100
        
        # Calculate liquidity scores
        bid_volume = sum([qty for price, qty in bids[:10]])
        ask_volume = sum([qty for price, qty in asks[:10]])
        total_volume = bid_volume + ask_volume
        
        # Imbalance ratio (>1 = more buying pressure, <1 = more selling pressure)
        imbalance_ratio = bid_volume / ask_volume if ask_volume > 0 else 1.0
        
        # Liquidity score (0-100, higher = better liquidity)
        liquidity_score = min(100, total_volume / 1000 * 10)  # Normalize based on volume
        
        return {
            'liquidity_score': liquidity_score,
            'imbalance_ratio': imbalance_ratio,
            'spread': spread_pct,
            'bid_volume': bid_volume,
            'ask_volume': ask_volume
        }
        
    except Exception as e:
        print(f"Error analyzing order book: {e}")
        return {'liquidity_score': 50, 'imbalance_ratio': 1.0, 'spread': 0}

# ========== NEW FEATURE 2: VOLUME PROFILE ANALYSIS ==========
def analyze_volume_profile(multi_tf_data):
    """Analyze volume profile to find key levels"""
    try:
        if '1h' not in multi_tf_data or 'volumes' not in multi_tf_data['1h']:
            return {'poc': 0, 'volume_trend': 'neutral', 'volume_spike': False}
        
        volumes = multi_tf_data['1h']['volumes']
        closes = multi_tf_data['1h']['closes']
        
        if len(volumes) < 10 or len(closes) < 10:
            return {'poc': 0, 'volume_trend': 'neutral', 'volume_spike': False}
        
        # Calculate volume-weighted average price (VWAP approximation)
        total_volume = sum(volumes[-20:])
        if total_volume == 0:
            return {'poc': 0, 'volume_trend': 'neutral', 'volume_spike': False}
        
        vwap = sum(closes[i] * volumes[i] for i in range(-20, 0)) / total_volume
        
        # Volume trend analysis
        recent_vol_avg = sum(volumes[-5:]) / 5
        older_vol_avg = sum(volumes[-15:-10]) / 5
        
        if recent_vol_avg > older_vol_avg * 1.3:
            volume_trend = 'increasing'
        elif recent_vol_avg < older_vol_avg * 0.7:
            volume_trend = 'decreasing'
        else:
            volume_trend = 'stable'
        
        # Volume spike detection
        latest_volume = volumes[-1]
        avg_volume = sum(volumes[-20:]) / 20
        volume_spike = latest_volume > avg_volume * 2
        
        return {
            'poc': vwap,  # Point of Control (simplified)
            'volume_trend': volume_trend,
            'volume_spike': volume_spike,
            'volume_ratio': recent_vol_avg / older_vol_avg if older_vol_avg > 0 else 1
        }
        
    except Exception as e:
        print(f"Error in volume profile analysis: {e}")
        return {'poc': 0, 'volume_trend': 'neutral', 'volume_spike': False}

# ========== NEW FEATURE 3: SOPHISTICATED TECHNICAL INDICATORS ==========
def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    if len(prices) < period:
        return {'upper': 0, 'middle': 0, 'lower': 0, 'squeeze': False}
    
    sma = sum(prices[-period:]) / period
    variance = sum((p - sma) ** 2 for p in prices[-period:]) / period
    std = variance ** 0.5
    
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    
    # Bollinger Band squeeze detection
    band_width = (upper - lower) / sma
    squeeze = band_width < 0.1  # Tight bands indicate low volatility
    
    return {
        'upper': upper,
        'middle': sma,
        'lower': lower,
        'squeeze': squeeze,
        'width': band_width
    }

def calculate_ichimoku(highs, lows, closes):
    """Calculate Ichimoku Cloud components"""
    if len(highs) < 52 or len(lows) < 52 or len(closes) < 52:
        return {'signal': 'neutral', 'cloud_direction': 'neutral'}
    
    # Tenkan-sen (9-period)
    tenkan = (max(highs[-9:]) + min(lows[-9:])) / 2
    
    # Kijun-sen (26-period)
    kijun = (max(highs[-26:]) + min(lows[-26:])) / 2
    
    # Senkou Span A (leading span A)
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (leading span B, 52-period)
    senkou_b = (max(highs[-52:]) + min(lows[-52:])) / 2
    
    current_price = closes[-1]
    
    # Signal generation
    if current_price > max(senkou_a, senkou_b) and tenkan > kijun:
        signal = 'bullish'
    elif current_price < min(senkou_a, senkou_b) and tenkan < kijun:
        signal = 'bearish'
    else:
        signal = 'neutral'
    
    # Cloud direction
    cloud_direction = 'bullish' if senkou_a > senkou_b else 'bearish'
    
    return {
        'signal': signal,
        'cloud_direction': cloud_direction,
        'tenkan': tenkan,
        'kijun': kijun
    }

def calculate_macd_with_divergence(prices, fast=12, slow=26, signal=9):
    """Calculate MACD with divergence detection"""
    if len(prices) < slow + signal:
        return {'signal': 'neutral', 'divergence': None, 'histogram': 0}
    
    # Calculate EMAs
    def ema(data, period):
        multiplier = 2 / (period + 1)
        ema_val = sum(data[:period]) / period
        for price in data[period:]:
            ema_val = (price - ema_val) * multiplier + ema_val
        return ema_val
    
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    
    macd_line = ema_fast - ema_slow
    
    # Signal line (EMA of MACD)
    macd_values = []
    for i in range(slow, len(prices)):
        subset = prices[:i+1]
        if len(subset) >= slow:
            fast_ema = ema(subset, fast)
            slow_ema = ema(subset, slow)
            macd_values.append(fast_ema - slow_ema)
    
    if len(macd_values) < signal:
        return {'signal': 'neutral', 'divergence': None, 'histogram': 0}
    
    signal_line = ema(macd_values, signal)
    histogram = macd_line - signal_line
    
    # Signal generation
    if macd_line > signal_line and histogram > 0:
        macd_signal = 'bullish'
    elif macd_line < signal_line and histogram < 0:
        macd_signal = 'bearish'
    else:
        macd_signal = 'neutral'
    
    # Simple divergence detection (price vs MACD highs/lows)
    divergence = None
    if len(prices) >= 10 and len(macd_values) >= 10:
        recent_price_high = max(prices[-5:])
        older_price_high = max(prices[-10:-5])
        recent_macd_high = max(macd_values[-5:])
        older_macd_high = max(macd_values[-10:-5])
        
        if recent_price_high > older_price_high and recent_macd_high < older_macd_high:
            divergence = 'bearish'
        elif recent_price_high < older_price_high and recent_macd_high > older_macd_high:
            divergence = 'bullish'
    
    return {
        'signal': macd_signal,
        'divergence': divergence,
        'histogram': histogram,
        'macd_line': macd_line
    }

# ========== NEW FEATURE 4: MOMENTUM AND VOLATILITY CLUSTERING ==========
def analyze_momentum_volatility(multi_tf_data):
    """Analyze momentum and volatility clustering"""
    try:
        if '1h' not in multi_tf_data or 'closes' not in multi_tf_data['1h']:
            return {'momentum_score': 0, 'volatility_cluster': False, 'momentum_direction': 'neutral'}
        
        closes = multi_tf_data['1h']['closes']
        if len(closes) < 20:
            return {'momentum_score': 0, 'volatility_cluster': False, 'momentum_direction': 'neutral'}
        
        # Calculate returns
        returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
        
        # Momentum score (rate of change acceleration)
        recent_returns = returns[-5:]
        older_returns = returns[-10:-5]
        
        momentum_score = (sum(recent_returns) / len(recent_returns)) - (sum(older_returns) / len(older_returns))
        momentum_direction = 'bullish' if momentum_score > 0 else 'bearish'
        
        # Volatility clustering (GARCH-like detection)
        volatilities = [abs(r) for r in returns[-20:]]
        recent_vol = sum(volatilities[-5:]) / 5
        avg_vol = sum(volatilities) / len(volatilities)
        
        volatility_cluster = recent_vol > avg_vol * 1.5
        
        # Normalize momentum score to 0-100
        momentum_score_normalized = min(100, max(0, (momentum_score + 0.02) * 2500))
        
        return {
            'momentum_score': momentum_score_normalized,
            'volatility_cluster': volatility_cluster,
            'momentum_direction': momentum_direction,
            'volatility_ratio': recent_vol / avg_vol if avg_vol > 0 else 1
        }
        
    except Exception as e:
        print(f"Error in momentum/volatility analysis: {e}")
        return {'momentum_score': 0, 'volatility_cluster': False, 'momentum_direction': 'neutral'}

# ========== NEW FEATURE 5: SOPHISTICATED EXIT LOGIC ==========
def calculate_trailing_stop(position_data, multi_tf_data, atr):
    """Calculate dynamic trailing stop based on market structure"""
    try:
        current_price = position_data['mark_price']
        entry_price = position_data['entry_price']
        direction = position_data['direction']
        
        # Calculate current P&L
        if direction == 'LONG':
            pnl_percent = (current_price - entry_price) / entry_price
        else:
            pnl_percent = (entry_price - current_price) / entry_price
        
        # Base trailing distance (ATR-based)
        base_trail_distance = atr * 1.5
        
        # Adjust based on profit level
        if pnl_percent > 0.05:  # 5%+ profit
            trail_multiplier = 0.8  # Tighter trailing
        elif pnl_percent > 0.03:  # 3%+ profit
            trail_multiplier = 1.0  # Normal trailing
        else:
            trail_multiplier = 1.2  # Looser trailing
        
        trail_distance = base_trail_distance * trail_multiplier
        
        # Calculate new stop based on direction
        if direction == 'LONG':
            new_stop = current_price - trail_distance
        else:
            new_stop = current_price + trail_distance
        
        return {
            'new_stop': new_stop,
            'trail_distance': trail_distance,
            'should_update': pnl_percent > 0.02  # Only trail when in profit
        }
        
    except Exception as e:
        print(f"Error calculating trailing stop: {e}")
        return {'new_stop': 0, 'trail_distance': 0, 'should_update': False}

def calculate_partial_profit_levels(position_data, multi_tf_data):
    """Calculate levels for partial profit taking"""
    try:
        entry_price = position_data['entry_price']
        current_price = position_data['mark_price']
        direction = position_data['direction']
        
        if direction == 'LONG':
            pnl_percent = (current_price - entry_price) / entry_price
        else:
            pnl_percent = (entry_price - current_price) / entry_price
        
        profit_levels = []
        
        # Define profit taking levels
        if pnl_percent >= 0.025:  # 2.5% profit
            profit_levels.append({
                'level': 0.025,
                'percentage': 25,  # Take 25% profit
                'reason': 'Initial profit taking'
            })
        
        if pnl_percent >= 0.05:   # 5% profit
            profit_levels.append({
                'level': 0.05,
                'percentage': 50,  # Take 50% of remaining
                'reason': 'Strong move confirmation'
            })
        
        if pnl_percent >= 0.08:   # 8% profit
            profit_levels.append({
                'level': 0.08,
                'percentage': 75,  # Take 75% of remaining
                'reason': 'Exceptional performance'
            })
        
        return profit_levels
        
    except Exception as e:
        print(f"Error calculating profit levels: {e}")
        return []

# ========== ENHANCED ANALYSIS INTEGRATION ==========
def get_comprehensive_analysis(symbol, market_data, multi_tf_data):
    """Get comprehensive technical analysis with weighted scoring"""
    coin = symbol.replace('USDT', '')
    current_price = market_data[coin]['price']
    
    analysis = {
        'overall_score': 0,
        'direction': 'neutral',
        'confidence': 0,
        'factors': {}
    }
    
    try:
        # 1. Timeframe Alignment (Weight: 25%)
        tf_alignment = analyze_timeframe_alignment(multi_tf_data)
        tf_score = 0
        if tf_alignment['aligned']:
            if tf_alignment['direction'] in ['strong_bullish', 'strong_bearish']:
                tf_score = 90
            elif tf_alignment['direction'] in ['bullish', 'bearish']:
                tf_score = 70
        else:
            tf_score = 30
        
        analysis['factors']['timeframe_alignment'] = {
            'score': tf_score,
            'weight': FACTOR_WEIGHTS['timeframe_alignment'],
            'direction': tf_alignment['direction'],
            'details': tf_alignment
        }
        
        # 2. Market Regime (Weight: 20%)
        regime_data = detect_market_regime(multi_tf_data, current_price)
        regime_score = regime_data['confidence']
        
        analysis['factors']['market_regime'] = {
            'score': regime_score,
            'weight': FACTOR_WEIGHTS['market_regime'],
            'regime': regime_data['regime'],
            'details': regime_data
        }
        
        # 3. Momentum Analysis (Weight: 15%)
        momentum_data = analyze_momentum_volatility(multi_tf_data)
        momentum_score = momentum_data['momentum_score']
        
        analysis['factors']['momentum'] = {
            'score': momentum_score,
            'weight': FACTOR_WEIGHTS['momentum'],
            'direction': momentum_data['momentum_direction'],
            'details': momentum_data
        }
        
        # 4. Volume Profile (Weight: 12%)
        volume_data = analyze_volume_profile(multi_tf_data)
        volume_score = 50  # Base score
        if volume_data['volume_trend'] == 'increasing':
            volume_score += 30
        elif volume_data['volume_spike']:
            volume_score += 20
        
        analysis['factors']['volume_profile'] = {
            'score': volume_score,
            'weight': FACTOR_WEIGHTS['volume_profile'],
            'trend': volume_data['volume_trend'],
            'details': volume_data
        }
        
        # 5. Technical Indicators (Weight: 10%)
        tech_score = 50  # Base score
        
        if '1h' in multi_tf_data and 'closes' in multi_tf_data['1h']:
            closes = multi_tf_data['1h']['closes']
            highs = multi_tf_data['1h']['highs']
            lows = multi_tf_data['1h']['lows']
            
            # Bollinger Bands
            bb_data = calculate_bollinger_bands(closes)
            if current_price > bb_data['upper']:
                tech_score += 10  # Breakout above upper band
            elif current_price < bb_data['lower']:
                tech_score -= 10  # Breakdown below lower band
            
            # Ichimoku
            ichimoku_data = calculate_ichimoku(highs, lows, closes)
            if ichimoku_data['signal'] == 'bullish':
                tech_score += 15
            elif ichimoku_data['signal'] == 'bearish':
                tech_score -= 15
            
            # MACD
            macd_data = calculate_macd_with_divergence(closes)
            if macd_data['signal'] == 'bullish':
                tech_score += 10
            elif macd_data['signal'] == 'bearish':
                tech_score -= 10
        
        analysis['factors']['technical_indicators'] = {
            'score': max(0, min(100, tech_score)),
            'weight': FACTOR_WEIGHTS['technical_indicators'],
            'details': {
                'bollinger': bb_data if 'bb_data' in locals() else None,
                'ichimoku': ichimoku_data if 'ichimoku_data' in locals() else None,
                'macd': macd_data if 'macd_data' in locals() else None
            }
        }
        
        # 6. Volatility Analysis (Weight: 8%)
        volatility_score = 50
        if momentum_data['volatility_cluster']:
            volatility_score += 20  # High volatility can mean opportunity
        
        analysis['factors']['volatility'] = {
            'score': volatility_score,
            'weight': FACTOR_WEIGHTS['volatility'],
            'cluster': momentum_data['volatility_cluster'],
            'details': momentum_data
        }
        
        # 7. Liquidity Analysis (Weight: 6%)
        liquidity_data = get_order_book_analysis(symbol)
        liquidity_score = liquidity_data['liquidity_score']
        
        analysis['factors']['liquidity'] = {
            'score': liquidity_score,
            'weight': FACTOR_WEIGHTS['liquidity'],
            'imbalance': liquidity_data['imbalance_ratio'],
            'details': liquidity_data
        }
        
        # Calculate weighted overall score
        total_weighted_score = 0
        for factor, data in analysis['factors'].items():
            total_weighted_score += data['score'] * data['weight']
        
        analysis['overall_score'] = total_weighted_score
        
        # Determine overall direction and confidence
        if total_weighted_score >= 65:
            analysis['direction'] = 'bullish'
            analysis['confidence'] = min(10, int((total_weighted_score - 50) / 5))
        elif total_weighted_score <= 35:
            analysis['direction'] = 'bearish'
            analysis['confidence'] = min(10, int((50 - total_weighted_score) / 5))
        else:
            analysis['direction'] = 'neutral'
            analysis['confidence'] = 3
        
        return analysis
        
    except Exception as e:
        print(f"Error in comprehensive analysis: {e}")
        return analysis

# [Keep all existing functions: place_futures_order, close_futures_position, etc.]
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
    """Set stop loss and take profit orders with proper price precision"""
    try:
        # Get position info
        positions = get_open_positions()
        if symbol not in positions:
            return None
        
        position = positions[symbol]
        quantity = abs(position['size'])
        
        # Round prices to proper precision based on coin
        coin = symbol.replace('USDT', '')
        if coin in ['BTC']:
            stop_price = round(stop_price, 1)
            take_profit_price = round(take_profit_price, 1)
        elif coin in ['ETH']:
            stop_price = round(stop_price, 2)
            take_profit_price = round(take_profit_price, 2)
        elif coin in ['SOL', 'BNB']:
            stop_price = round(stop_price, 2)
            take_profit_price = round(take_profit_price, 2)
        else:
            stop_price = round(stop_price, 4)
            take_profit_price = round(take_profit_price, 4)
        
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

# ========== ENHANCED SMART PROFIT TAKING ==========
def check_profit_taking_opportunity(position_data, multi_tf_data):
    """Enhanced profit taking with sophisticated exit logic"""
    
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
    
    exit_signals = []
    signal_strength = 0
    
    # Get comprehensive analysis for exit decision
    symbol = position_data['symbol']
    coin = position_data['coin']
    
    # Create mock market_data for analysis
    market_data = {coin: {'price': current_price}}
    analysis = get_comprehensive_analysis(symbol, market_data, multi_tf_data)
    
    # 1. Check if overall market sentiment changed
    if direction == 'LONG' and analysis['direction'] == 'bearish' and analysis['confidence'] >= 6:
        exit_signals.append("Market turned bearish")
        signal_strength += 3
    elif direction == 'SHORT' and analysis['direction'] == 'bullish' and analysis['confidence'] >= 6:
        exit_signals.append("Market turned bullish") 
        signal_strength += 3
    
    # 2. Technical indicator exits
    if '1h' in multi_tf_data and 'closes' in multi_tf_data['1h']:
        closes = multi_tf_data['1h']['closes']
        
        # RSI extreme levels
        current_rsi = calculate_rsi(closes)
        if direction == 'LONG' and current_rsi > 78:
            exit_signals.append(f"RSI extremely overbought ({current_rsi:.0f})")
            signal_strength += 2
        elif direction == 'SHORT' and current_rsi < 22:
            exit_signals.append(f"RSI extremely oversold ({current_rsi:.0f})")
            signal_strength += 2
        
        # Bollinger Band extremes
        bb_data = calculate_bollinger_bands(closes)
        if direction == 'LONG' and current_price > bb_data['upper'] * 1.02:
            exit_signals.append("Price extended beyond upper Bollinger Band")
            signal_strength += 1
        elif direction == 'SHORT' and current_price < bb_data['lower'] * 0.98:
            exit_signals.append("Price extended beyond lower Bollinger Band")
            signal_strength += 1
    
    # 3. Volume divergence
    volume_data = analyze_volume_profile(multi_tf_data)
    if volume_data['volume_trend'] == 'decreasing' and pnl_percent > 0.03:
        exit_signals.append("Volume declining on move")
        signal_strength += 1
    
    # 4. Momentum loss
    momentum_data = analyze_momentum_volatility(multi_tf_data)
    if ((direction == 'LONG' and momentum_data['momentum_direction'] == 'bearish') or 
        (direction == 'SHORT' and momentum_data['momentum_direction'] == 'bullish')):
        exit_signals.append("Momentum turning against position")
        signal_strength += 2
    
    # 5. Profit level based exits
    partial_levels = calculate_partial_profit_levels(position_data, multi_tf_data)
    if partial_levels:
        exit_signals.append(f"Reached profit target ({pnl_percent*100:.1f}%)")
        signal_strength += 1
    
    # Decision logic: More sophisticated than before
    should_exit = False
    exit_type = "hold"
    
    if signal_strength >= 4 and pnl_percent > 0.02:
        should_exit = True
        exit_type = "full_exit"
    elif signal_strength >= 2 and pnl_percent > 0.04:
        should_exit = True
        exit_type = "partial_exit"
    elif pnl_percent > 0.08:  # 8%+ profit, always take some
        should_exit = True
        exit_type = "partial_exit"
    
    if should_exit:
        reason = f"Smart {exit_type}: {', '.join(exit_signals)} (Profit: {pnl_percent*100:.1f}%)"
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
    """Enhanced AI trading decision with comprehensive analysis and weighted scoring"""
    if not client:
        return None
    
    current_price = market_data[coin]['price']
    symbol = f"{coin}USDT"
    
    # Get comprehensive analysis with weighted factors
    analysis = get_comprehensive_analysis(symbol, market_data, multi_tf_data)
    
    # Skip if overall score is too neutral
    if 40 <= analysis['overall_score'] <= 60:
        return {'direction': 'SKIP', 'reason': 'Market conditions too neutral'}
    
    # Calculate advanced indicators for stop/target calculation
    hourly_data = multi_tf_data.get('1h', {})
    
    atr = 0
    if 'highs' in hourly_data and 'lows' in hourly_data and 'closes' in hourly_data:
        atr = calculate_atr(hourly_data['highs'], hourly_data['lows'], hourly_data['closes'])
    
    market_regime_info = analysis['factors']['market_regime']['details']
    
    dynamic_stops = calculate_dynamic_stops(
        current_price, 
        atr, 
        'LONG' if analysis['direction'] == 'bullish' else 'SHORT',
        market_regime_info['regime']
    )
    
    # Get AI memory context
    ai_context = get_ai_context_memory()
    
    # Build detailed factor summary for AI
    factor_summary = "WEIGHTED FACTOR ANALYSIS:\n"
    for factor_name, factor_data in analysis['factors'].items():
        weight_pct = factor_data['weight'] * 100
        contribution = factor_data['score'] * factor_data['weight']
        factor_summary += f"• {factor_name.upper()}: {factor_data['score']:.0f}/100 (Weight: {weight_pct:.0f}%) = {contribution:.1f}\n"
    
    factor_summary += f"\nOVERALL WEIGHTED SCORE: {analysis['overall_score']:.1f}/100\n"
    factor_summary += f"DIRECTION: {analysis['direction'].upper()} (Confidence: {analysis['confidence']}/10)\n"
    
    # Enhanced prompt with comprehensive analysis
    analysis_prompt = f"""
{ai_context}

---

{coin} COMPREHENSIVE MARKET ANALYSIS

PRICE DATA:
- Current: ${current_price:.2f}
- 24h Change: {market_data[coin]['change_24h']:+.1f}%
- Volume: {market_data[coin]['volume']/1000000:.1f}M

{factor_summary}

KEY INSIGHTS:
- Market Regime: {market_regime_info['regime']} ({market_regime_info['confidence']:.0f}% confidence)
- Timeframe Alignment: {analysis['factors']['timeframe_alignment']['direction']}
- Momentum: {analysis['factors']['momentum']['direction']} ({analysis['factors']['momentum']['score']:.0f}/100)
- Volume Trend: {analysis['factors']['volume_profile']['trend']}
- Liquidity Score: {analysis['factors']['liquidity']['score']:.0f}/100
- Technical Indicators: {analysis['factors']['technical_indicators']['score']:.0f}/100

RISK MANAGEMENT:
- Suggested SL: {dynamic_stops['sl_percentage']:.1f}%
- Suggested TP: {dynamic_stops['tp_percentage']:.1f}%
- Risk/Reward: 1:{dynamic_stops['risk_reward']:.1f}
- ATR: ${atr:.2f}

HISTORICAL PERFORMANCE:
- Win Rate: {learning_insights.get('win_rate', 0):.0f}%
- Best Leverage: {learning_insights.get('best_leverage', 15)}x

TRADING RULES:
1. Only trade when weighted score > 65 (bullish) or < 35 (bearish)
2. Require at least 2:1 risk/reward ratio
3. Confidence should reflect score strength and factor alignment
4. Higher confidence = higher leverage (10-30x range)
5. Position size 5-15% based on confidence and volatility

Based on the comprehensive weighted analysis, provide:
DECISION: [LONG/SHORT/SKIP]
LEVERAGE: [10-30] (scale with confidence)
SIZE: [5-15]%
CONFIDENCE: [1-10] (must align with weighted score)
REASON: [key factors driving decision]
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": analysis_prompt}],
            max_tokens=200,
            temperature=0.3  # Lower temperature for more consistent decisions
        )
        
        track_cost('openai', 0.00012, '500')  # Slightly higher cost due to longer prompt
        
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
                    leverage_str = ''.join(filter(str.isdigit, value))
                    trade_params['leverage'] = min(30, max(10, int(leverage_str) if leverage_str else 15))
                elif 'SIZE' in key:
                    size_str = ''.join(filter(lambda x: x.isdigit() or x == '.', value))
                    if size_str:
                      size = float(size_str)
                      trade_params['position_size'] = min(0.15, max(0.05, size / 100))
                elif 'CONFIDENCE' in key:
                    conf_str = ''.join(filter(str.isdigit, value))
                    trade_params['confidence'] = min(10, max(1, int(conf_str) if conf_str else 5))
                elif 'REASON' in key:
                    trade_params['reasoning'] = value[:200]
        
        # Add analysis results to trade params
        trade_params['stop_loss'] = dynamic_stops['stop_loss']
        trade_params['take_profit'] = dynamic_stops['take_profit']
        trade_params['sl_percentage'] = dynamic_stops['sl_percentage']
        trade_params['tp_percentage'] = dynamic_stops['tp_percentage']
        trade_params['market_regime'] = market_regime_info['regime']
        trade_params['atr'] = atr
        trade_params['overall_score'] = analysis['overall_score']
        trade_params['analysis_direction'] = analysis['direction']
        
        # Default values with score-based adjustments
        if 'direction' not in trade_params:
            trade_params['direction'] = 'SKIP'
        if 'leverage' not in trade_params:
            # Scale leverage with confidence/score
            if analysis['overall_score'] > 75:
                trade_params['leverage'] = 25
            elif analysis['overall_score'] > 65:
                trade_params['leverage'] = 20
            else:
                trade_params['leverage'] = 15
        if 'position_size' not in trade_params:
            # Use dynamic position sizing based on portfolio state
            positions = get_open_positions()
            sizing_data = get_dynamic_position_size(len(positions), analysis['confidence'], balance)
            trade_params['position_size'] = sizing_data['position_size']
        if 'confidence' not in trade_params:
            trade_params['confidence'] = analysis['confidence']
        if 'reasoning' not in trade_params:
            top_factors = sorted(analysis['factors'].items(), 
                               key=lambda x: x[1]['score'] * x[1]['weight'], reverse=True)[:2]
            trade_params['reasoning'] = f"Top factors: {', '.join([f[0] for f in top_factors])}"
        
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
    """Execute real trade on Binance futures with enhanced balance management"""
    if trade_params['direction'] == 'SKIP':
        return
    
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        symbol = f"{coin}USDT"
        
        # Enhanced portfolio risk check
        portfolio_risk = calculate_portfolio_risk()
        if not portfolio_risk['can_trade']:
            print(f"❌ Cannot trade: {portfolio_risk['reason']}")
            return
        
        # Enhanced pre-trade validation
        if trade_params.get('overall_score', 50) < 35 and trade_params['direction'] == 'SHORT':
            print(f"⚠️ Score too low for SHORT: {trade_params.get('overall_score', 50):.1f}")
            return
        elif trade_params.get('overall_score', 50) > 65 and trade_params['direction'] == 'LONG':
            print(f"⚠️ Score too low for LONG: {trade_params.get('overall_score', 50):.1f}")
            return
        
        balance = portfolio_risk['balance']
        current_positions = get_open_positions()
        
        # Use dynamic position sizing
        sizing_data = get_dynamic_position_size(len(current_positions), trade_params['confidence'], balance)
        position_value = sizing_data['position_value']
        leverage = trade_params['leverage']
        
        # Enhanced position size validation
        if position_value < 5:  # Minimum $5 position
            print(f"❌ Position value too small: ${position_value:.2f}")
            return
        
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
        
        print(f"   📊 Multi-Position Trade Analysis:")
        print(f"      Position #{len(current_positions) + 1}/6")
        print(f"      Dynamic Size: {sizing_data['position_size']*100:.1f}% (${position_value:.2f})")
        print(f"      Available Balance: ${sizing_data['available_balance']:.2f}")
        print(f"      Overall Score: {trade_params.get('overall_score', 50):.1f}/100")
        print(f"      Quantity: {quantity} {coin}")
        print(f"      Notional: ${quantity * current_price:.2f}")
        print(f"      Portfolio Margin Usage: {portfolio_risk['margin_ratio']*100:.1f}%")
        
        # Place order
        side = "BUY" if trade_params['direction'] == 'LONG' else "SELL"
        
        order_result = place_futures_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            leverage=leverage
        )
        
        if order_result:
            print(f"\n✅ MULTI-POSITION TRADE EXECUTED:")
            print(f"   Position #{len(current_positions) + 1}: {trade_params['direction']} {coin}")
            print(f"   Quantity: {quantity} @ {leverage}x")
            print(f"   Value: ${position_value:.2f} | Notional: ${notional_value:.2f}")
            print(f"   Confidence: {trade_params['confidence']}/10 | Score: {trade_params.get('overall_score', 50):.1f}")
            print(f"   Portfolio: {len(current_positions) + 1}/{MAX_CONCURRENT_POSITIONS} positions")
            print(f"   Reason: {trade_params['reasoning'][:60]}...")
            
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
    """Enhanced position monitoring with sophisticated exit logic"""
    try:
        positions = get_open_positions()
        
        for symbol, position_data in positions.items():
            coin = position_data['coin']
            
            # Get multi-timeframe data for analysis
            multi_tf_data = get_multi_timeframe_data(symbol)
            
            # Calculate ATR for trailing stops
            atr = 0
            if ('1h' in multi_tf_data and 'highs' in multi_tf_data['1h'] and 
                'lows' in multi_tf_data['1h'] and 'closes' in multi_tf_data['1h']):
                atr = calculate_atr(
                    multi_tf_data['1h']['highs'], 
                    multi_tf_data['1h']['lows'], 
                    multi_tf_data['1h']['closes']
                )
            
            # Check for enhanced smart exit opportunities
            should_close, reason = check_profit_taking_opportunity(position_data, multi_tf_data)
            
            if should_close:
                print(f"🧠 Enhanced exit triggered for {coin}: {reason}")
                
                # Determine if full or partial exit
                if "partial_exit" in reason:
                    # Implement partial exit logic here if needed
                    print(f"   📊 Partial exit recommended (not implemented yet)")
                
                close_result = close_futures_position(symbol)
                
                if close_result:
                    print(f"✅ Position closed: {coin}")
                    
                    # Update AI decision outcome
                    pnl_percent = (position_data['pnl'] / (position_data['notional'] / position_data['leverage'])) * 100
                    update_ai_decision_outcome(coin, position_data['direction'], pnl_percent)
                else:
                    print(f"❌ Failed to close position: {coin}")
            else:
                # Check for trailing stop updates
                if atr > 0:
                    trailing_data = calculate_trailing_stop(position_data, multi_tf_data, atr)
                    if trailing_data['should_update']:
                        print(f"   📈 {coin} trailing stop update available: {trailing_data['new_stop']:.2f}")
                        # Could implement automatic trailing stop updates here
        
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

def calculate_portfolio_value(market_data):
    """Calculate total portfolio value from real Binance data"""
    balance = get_futures_balance()
    positions = get_open_positions()
    
    total_pnl = sum(pos['pnl'] for pos in positions.values())
    
    return balance + total_pnl

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

def calculate_portfolio_risk():
    """Calculate current portfolio risk and available capacity"""
    try:
        positions = get_open_positions()
        balance = get_futures_balance()
        
        if balance <= 0:
            return {'can_trade': False, 'reason': 'No balance available'}
        
        # Calculate total exposure
        total_notional = sum(pos['notional'] for pos in positions.values())
        total_margin_used = sum(pos['notional'] / pos['leverage'] for pos in positions.values())
        
        # Risk metrics
        portfolio_risk = {
            'positions_count': len(positions),
            'total_notional': total_notional,
            'total_margin_used': total_margin_used,
            'balance': balance,
            'margin_ratio': total_margin_used / balance if balance > 0 else 0,
            'available_balance': balance - total_margin_used,
            'max_positions': MAX_CONCURRENT_POSITIONS,
            'can_trade': True,
            'reason': 'Good to trade'
        }
        
        # Risk checks
        if len(positions) >= MAX_CONCURRENT_POSITIONS:
            portfolio_risk['can_trade'] = False
            portfolio_risk['reason'] = f'Maximum positions reached ({len(positions)}/{MAX_CONCURRENT_POSITIONS})'
        elif portfolio_risk['margin_ratio'] > 0.8:  # 80% margin utilization limit
            portfolio_risk['can_trade'] = False
            portfolio_risk['reason'] = f'High margin utilization ({portfolio_risk["margin_ratio"]*100:.1f}%)'
        elif portfolio_risk['available_balance'] < 5:  # Minimum $5 required
            portfolio_risk['can_trade'] = False
            portfolio_risk['reason'] = f'Insufficient available balance (${portfolio_risk["available_balance"]:.2f})'
        
        return portfolio_risk
        
    except Exception as e:
        print(f"Error calculating portfolio risk: {e}")
        return {'can_trade': False, 'reason': f'Risk calculation error: {e}'}

def get_dynamic_position_size(current_positions_count, confidence_score, balance):
    """Calculate dynamic position size based on current portfolio and confidence"""
    try:
        # Base position size from scaling table
        base_size = POSITION_SCALING.get(current_positions_count, 0.04)  # Default 4% for 7th+ positions
        
        # Adjust based on confidence (6-10 range)
        confidence_multiplier = 0.7 + (confidence_score - 6) * 0.075  # 0.7 to 1.0 multiplier
        
        # Adjust based on available balance
        positions = get_open_positions()
        used_margin = sum(pos['notional'] / pos['leverage'] for pos in positions.values())
        available_balance = balance - used_margin
        
        # Ensure we don't use more than 90% of available balance
        max_position_value = available_balance * 0.9
        
        # Calculate final position size
        target_position_value = balance * base_size * confidence_multiplier
        final_position_size = min(target_position_value, max_position_value) / balance
        
        # Ensure minimum and maximum bounds
        final_position_size = max(0.03, min(0.15, final_position_size))  # 3% to 15% range
        
        return {
            'position_size': final_position_size,
            'base_size': base_size,
            'confidence_multiplier': confidence_multiplier,
            'available_balance': available_balance,
            'position_value': balance * final_position_size
        }
        
    except Exception as e:
        print(f"Error calculating dynamic position size: {e}")
        return {'position_size': 0.05, 'position_value': balance * 0.05}

def run_enhanced_bot():
    """Main bot loop with enhanced analysis and intelligent decision making"""
    init_database()
    
    print("🚀 ENHANCED SMART TRADING BOT - LIVE MODE")
    print("🧠 Features: Weighted Factor Analysis + Advanced Technical Indicators")
    print("📊 Analysis: Order Book + Volume Profile + Ichimoku + Bollinger + MACD")
    print("⚡ Exits: Sophisticated Profit Taking + Trailing Stops + Partial Exits")
    print("🎯 Intelligence: Weighted Scoring System + Factor Importance Ranking")
    print("💰 Multi-Position: Up to 6 concurrent trades with dynamic sizing")
    print("📈 Balance Management: Intelligent position scaling (15% → 5%)")
    print("🔄 Risk Control: 80% max margin usage + dynamic balance allocation")
    print("📱 Coins: BTC, ETH, SOL, BNB, SEI, DOGE, TIA, TAO, ARB, SUI, ENA, FET")
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
            
            # Enhanced position monitoring
            current_positions = monitor_positions()
            
            # Full analysis every 3 minutes with enhanced intelligence
            if current_time - last_full_analysis >= 180:  # 3 minutes = 180 seconds
                print(f"\n{'='*80}")
                print(f"🧠 ENHANCED MARKET ANALYSIS #{iteration}")
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
                
                # Enhanced market overview
                print(f"\n💰 Account Status:")
                print(f"   Balance: ${current_balance:.2f}")
                print(f"   Open Positions: {len(current_positions)}")
                
                if current_positions:
                    total_pnl = sum(pos['pnl'] for pos in current_positions.values())
                    print(f"   Unrealized P&L: ${total_pnl:+.2f}")
                
                print(f"\n📊 Enhanced Market Overview:")
                for coin in target_coins[:6]:
                    if coin in market_data:
                        data = market_data[coin]
                        symbol = f"{coin}USDT"
                        coin_tf_data = get_multi_timeframe_data(symbol)
                        
                        # Get comprehensive analysis for overview
                        analysis = get_comprehensive_analysis(symbol, {coin: data}, coin_tf_data)
                        
                        # Market regime indicators
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
                        
                        # Enhanced display with weighted score
                        score = analysis['overall_score']
                        score_emoji = "🟢" if score > 65 else "🔴" if score < 35 else "🟡"
                        
                        print(f"   {coin}: ${data['price']:.2f} {trend} ({data['change_24h']:+.1f}%) "
                              f"Score:{score:.0f} {score_emoji}{position_info}")
                
                # Look for new opportunities with enhanced intelligence and multi-position support
                portfolio_risk = calculate_portfolio_risk()
                
                if portfolio_risk['can_trade']:
                    learning_insights = get_learning_insights()
                    
                    print(f"\n🤖 Multi-Position AI Analysis & Opportunity Scan:")
                    print(f"   Portfolio: {portfolio_risk['positions_count']}/{MAX_CONCURRENT_POSITIONS} positions")
                    print(f"   Available Balance: ${portfolio_risk['available_balance']:.2f}")
                    print(f"   Margin Usage: {portfolio_risk['margin_ratio']*100:.1f}%")
                    print(f"   Historical Win Rate: {learning_insights['win_rate']:.1f}%")
                    print(f"   Analyzing with weighted factor system...")
                    
                    # Get AI context for decision making
                    ai_context = get_ai_context_memory()
                    if "win rate" in ai_context.lower():
                        recent_performance = ai_context.split("RECENT PERFORMANCE:")[1].split("\n")[0].strip() if "RECENT PERFORMANCE:" in ai_context else ""
                        if recent_performance:
                            print(f"   Recent AI Performance: {recent_performance}")
                    
                    opportunities_found = 0
                    opportunities_analyzed = 0
                    max_new_positions = min(3, MAX_CONCURRENT_POSITIONS - len(current_positions))  # Max 3 new positions per cycle
                    
                    # Score all coins first, then pick the best ones
                    coin_scores = []
                    
                    for coin in target_coins:
                        # Skip if already have position
                        symbol_check = f"{coin}USDT"
                        if symbol_check in current_positions:
                            continue
                        
                        opportunities_analyzed += 1
                        print(f"   🔍 Analyzing {coin} with enhanced factors...")
                        
                        symbol = f"{coin}USDT"
                        multi_tf_data = get_multi_timeframe_data(symbol)
                        
                        # Get comprehensive weighted analysis
                        analysis = get_comprehensive_analysis(symbol, market_data, multi_tf_data)
                        
                        print(f"      Weighted Score: {analysis['overall_score']:.1f}/100")
                        print(f"      Direction: {analysis['direction']} (Confidence: {analysis['confidence']}/10)")
                        
                        # Top factors analysis
                        top_factors = sorted(analysis['factors'].items(), 
                                           key=lambda x: x[1]['score'] * x[1]['weight'], reverse=True)[:2]
                        print(f"      Top Factors: {', '.join([f[0] for f in top_factors])}")
                        
                        if analysis['overall_score'] > 65 or analysis['overall_score'] < 35:
                            coin_scores.append((coin, analysis['overall_score'], analysis))
                            print(f"      ✅ Qualified for trading consideration")
                        else:
                            print(f"      ⏭️ Score too neutral ({analysis['overall_score']:.1f})")
                    
                    # Pick the best opportunities (multiple trades possible)
                    if coin_scores:
                        # Sort by score distance from 50 (most extreme = best)
                        coin_scores.sort(key=lambda x: abs(x[1] - 50), reverse=True)
                        
                        print(f"\n   🎯 QUALIFIED OPPORTUNITIES: {len(coin_scores)} coins")
                        
                        # Process up to max_new_positions opportunities
                        for i, (coin, score, analysis) in enumerate(coin_scores[:max_new_positions]):
                            if opportunities_found >= max_new_positions:
                                break
                                
                            print(f"\n   📊 OPPORTUNITY #{i+1}: {coin}")
                            print(f"      Score: {score:.1f}/100 | Direction: {analysis['direction']}")
                            
                            # Run AI decision on this candidate
                            symbol = f"{coin}USDT"
                            multi_tf_data = get_multi_timeframe_data(symbol)
                            
                            print(f"      🧠 Running enhanced AI analysis...")
                            trade_params = ai_trade_decision(coin, market_data, multi_tf_data, learning_insights)
                            
                            if trade_params and trade_params['direction'] != 'SKIP':
                                print(f"      AI Decision: {trade_params['direction']} (Confidence: {trade_params['confidence']}/10)")
                                print(f"      Reasoning: {trade_params['reasoning'][:80]}...")
                                
                                if trade_params['confidence'] >= MIN_CONFIDENCE_THRESHOLD:
                                    risk_reward = trade_params.get('tp_percentage', 0) / trade_params.get('sl_percentage', 1)
                                    print(f"      Risk/Reward: 1:{risk_reward:.1f}")
                                    
                                    # Dynamic position sizing info
                                    sizing_data = get_dynamic_position_size(len(current_positions) + opportunities_found, 
                                                                          trade_params['confidence'], 
                                                                          portfolio_risk['balance'])
                                    print(f"      Dynamic Size: {sizing_data['position_size']*100:.1f}% (${sizing_data['position_value']:.2f})")
                                    
                                    if risk_reward >= 1.8:  # Maintain risk/reward threshold
                                        print(f"      ✅ EXECUTING MULTI-POSITION TRADE!")
                                        print(f"         Position #{len(current_positions) + opportunities_found + 1}: {trade_params['direction']} {coin}")
                                        print(f"         Score: {score:.1f} | Leverage: {trade_params['leverage']}x")
                                        
                                        execute_real_trade(coin, trade_params, market_data[coin]['price'])
                                        opportunities_found += 1
                                        
                                        # Brief wait between trades
                                        time.sleep(1)
                                    else:
                                        print(f"      ⏭️ Risk/reward too low (1:{risk_reward:.1f})")
                                else:
                                    print(f"      ⏭️ AI confidence too low ({trade_params['confidence']}/10)")
                            else:
                                print(f"      ⏭️ AI recommends SKIP despite good score")
                    
                    if opportunities_found == 0:
                        print(f"   📊 Multi-Position Analysis Complete: No suitable opportunities from {opportunities_analyzed} coins")
                        print(f"      Waiting for higher-quality setups...")
                    else:
                        print(f"   🎯 Executed {opportunities_found} trades | Portfolio: {len(current_positions) + opportunities_found}/{MAX_CONCURRENT_POSITIONS}")
                else:
                    print(f"\n⚠️ Cannot open new positions: {portfolio_risk['reason']}")
                    print(f"   Current: {portfolio_risk['positions_count']}/{MAX_CONCURRENT_POSITIONS} positions")
                    print(f"   Margin Usage: {portfolio_risk['margin_ratio']*100:.1f}%")
                    print(f"   Available Balance: ${portfolio_risk['available_balance']:.2f}")
                
                # Enhanced performance summary with multi-position metrics
                total_value = calculate_portfolio_value(market_data)
                cost_data = get_cost_projections()
                final_portfolio_risk = calculate_portfolio_risk()
                
                print(f"\n📈 Multi-Position Performance Summary:")
                print(f"   Total Value: ${total_value:.2f}")
                print(f"   Active Positions: {final_portfolio_risk['positions_count']}/{MAX_CONCURRENT_POSITIONS}")
                print(f"   Portfolio Margin Usage: {final_portfolio_risk['margin_ratio']*100:.1f}%")
                print(f"   Available for New Trades: ${final_portfolio_risk['available_balance']:.2f}")
                print(f"   Session Costs: ${cost_data['current']['total']:.4f}")
                print(f"   Analysis Quality: Enhanced with {len(FACTOR_WEIGHTS)} weighted factors")
                
                last_full_analysis = current_time
                print("="*80)
            
            time.sleep(QUICK_CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            print("\n\n🛑 Enhanced bot stopped by user")
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
    
    # Start enhanced trading bot
    run_enhanced_bot()