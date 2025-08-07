import requests
import time
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_btc_price():
    url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
    response = requests.get(url)
    return float(response.json()['price'])

def ai_should_buy(current_price, price_history):
    if not client.api_key:
        return "WAIT (No API key)"
    
    prompt = f"""
    BTC current price: ${current_price:,.2f}
    Recent prices: {price_history}
    
    Should I buy, sell, or wait? Answer with just: BUY, SELL, or WAIT
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10
    )
    
    return response.choices[0].message.content.strip()

price_history = []

while True:
    price = get_btc_price()
    price_history.append(price)
    
    if len(price_history) > 5:
        price_history = price_history[-5:]
    
    if len(price_history) >= 3:
        decision = ai_should_buy(price, price_history)
        print(f"BTC: ${price:,.2f} | AI says: {decision}")
    else:
        print(f"BTC: ${price:,.2f} | Building history...")
    
    time.sleep(30)