import requests
import time

def get_railway_ip():
    """Get Railway deployment IP for Binance API restriction"""
    print("🚀 RAILWAY IP DETECTOR")
    print("=" * 60)
    
    services = [
        ("ipify", "https://api.ipify.org"),
        ("httpbin", "https://httpbin.org/ip"),
        ("myip", "https://api.myip.com"),
        ("ipapi", "https://ipapi.co/ip")
    ]
    
    found_ip = None
    
    for name, url in services:
        try:
            print(f"📡 Trying {name}...")
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                if name == "httpbin":
                    ip = response.json()['origin']
                elif name == "myip":
                    ip = response.json()['ip']
                else:
                    ip = response.text.strip()
                
                print(f"✅ {name}: {ip}")
                found_ip = ip
                break
                
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            continue
    
    if found_ip:
        print("=" * 60)
        print("🎉 SUCCESS!")
        print(f"🌐 YOUR RAILWAY IP: {found_ip}")
        print("=" * 60)
        print("📋 COPY THIS IP TO BINANCE:")
        print(f"   {found_ip}")
        print("=" * 60)
        print("🔧 BINANCE SETUP:")
        print("1. Binance.com → Profile → API Management")
        print("2. Edit your API key")
        print("3. Add this IP to restrictions:")
        print(f"   {found_ip}")
        print("4. Enable 'Enable Futures' ✅")
        print("5. Enable 'Enable Trading' ✅")
        print("6. Save")
        print("=" * 60)
        
        # Keep showing the IP for 5 minutes so you can copy it
        for i in range(30):
            print(f"🌐 YOUR IP: {found_ip} (showing for {30-i} more times)")
            time.sleep(10)
            
    else:
        print("❌ Could not get IP - check Railway logs")
    
    return found_ip

if __name__ == "__main__":
    get_railway_ip()