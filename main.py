import requests
import time

def get_btc_price():
    url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
    response = requests.get(url)
    price = response.json()['price']
    return float(price)

# Test it works
while True:
    price = get_btc_price()
    print(f"BTC Price: ${price:,.2f}")
    time.sleep(10)  # Check every 10 seconds