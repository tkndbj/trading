#!/usr/bin/env python3
"""
Simple script to get Railway deployment IP address for Binance API restriction
Upload this as get_ip.py to your Railway project and run it once
"""

import requests
import json

def get_public_ip():
    """Get the public IP address of this Railway deployment"""
    services = [
        "https://api.ipify.org?format=json",
        "https://httpbin.org/ip", 
        "https://api.myip.com",
        "https://ipapi.co/json"
    ]
    
    print("🔍 Getting Railway deployment IP address...")
    print("=" * 50)
    
    for i, service in enumerate(services, 1):
        try:
            print(f"📡 Trying service {i}...")
            response = requests.get(service, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Different services return IP in different formats
                ip = None
                if 'ip' in data:
                    ip = data['ip']
                elif 'origin' in data:
                    ip = data['origin']
                elif 'query' in data:
                    ip = data['query']
                
                if ip:
                    print(f"✅ SUCCESS!")
                    print(f"🌐 Your Railway IP: {ip}")
                    print("=" * 50)
                    print("📋 COPY THIS IP TO BINANCE:")
                    print(f"   {ip}")
                    print("=" * 50)
                    print("🔧 Binance Setup Steps:")
                    print("1. Go to Binance.com → Profile → API Management")
                    print("2. Click 'Edit' on your API key")
                    print("3. In 'API restrictions', add this IP:")
                    print(f"   {ip}")
                    print("4. Enable 'Enable Futures' ✅")
                    print("5. Enable 'Enable Trading' ✅")
                    print("6. Save changes")
                    print("=" * 50)
                    return ip
                    
        except Exception as e:
            print(f"❌ Service {i} failed: {e}")
            continue
    
    print("❌ Could not determine IP address")
    print("💡 Try running this script again or check Railway logs")
    return None

def test_ip_services():
    """Test multiple IP services and show all results"""
    print("\n🧪 TESTING ALL IP SERVICES:")
    print("=" * 50)
    
    services = {
        "ipify": "https://api.ipify.org",
        "httpbin": "https://httpbin.org/ip",
        "myip": "https://api.myip.com", 
        "ipapi": "https://ipapi.co/ip"
    }
    
    results = {}
    
    for name, url in services.items():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                if 'json' in url or name == 'httpbin':
                    try:
                        data = response.json()
                        ip = data.get('ip', data.get('origin', 'unknown'))
                    except:
                        ip = response.text.strip()
                else:
                    ip = response.text.strip()
                
                results[name] = ip
                print(f"✅ {name:8}: {ip}")
            else:
                print(f"❌ {name:8}: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ {name:8}: {str(e)[:30]}")
    
    # Find most common IP (in case of differences)
    if results:
        ips = list(results.values())
        most_common = max(set(ips), key=ips.count)
        
        print("=" * 50)
        print(f"🎯 MOST LIKELY IP: {most_common}")
        print(f"📊 Consistency: {ips.count(most_common)}/{len(ips)} services agree")
        
        if ips.count(most_common) == len(ips):
            print("✅ ALL SERVICES AGREE - This is definitely your IP!")
        else:
            print("⚠️ Some inconsistency - use the most common one above")
            
        return most_common
    
    return None

if __name__ == "__main__":
    print("🚀 RAILWAY IP DETECTOR")
    print("=" * 50)
    print("This will find your Railway deployment's public IP")
    print("for Binance API restriction setup.")
    print("")
    
    # Get IP using primary method
    ip = get_public_ip()
    
    if not ip:
        print("\n🔄 Trying alternative method...")
        ip = test_ip_services()
    
    if ip:
        print(f"\n🎉 SUCCESS! Your Railway IP is: {ip}")
        print("\n📝 Next steps:")
        print("1. Copy this IP")
        print("2. Go to Binance API settings") 
        print("3. Add this IP to restrictions")
        print("4. Enable Futures trading")
        print("5. Run your trading bot!")
        
        # Save IP to a file for reference
        try:
            with open('railway_ip.txt', 'w') as f:
                f.write(f"Railway IP: {ip}\nTimestamp: {requests.get('http://worldtimeapi.org/api/timezone/UTC').json()['datetime']}\n")
            print("\n💾 IP saved to 'railway_ip.txt' for your records")
        except:
            pass
            
    else:
        print("\n❌ Could not determine IP address")
        print("💡 Alternative: Check Railway logs when your bot runs")
        print("   Your bot will show API connection errors with IP info")
        
    print(f"\n⏰ Script completed")