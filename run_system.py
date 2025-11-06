"""
🍽️ Food Detection System - All-in-One Runner
Cara gampang menjalankan sistem deteksi makanan Indonesia
"""

import subprocess
import sys
import time
import webbrowser
import os
from threading import Thread

def print_banner():
    """Print system banner"""
    print("=" * 60)
    print("🍽️  FOOD DETECTION SYSTEM - INDONESIAN FOOD")
    print("=" * 60)
    print("📍 Backend: http://localhost:5000")
    print("🌐 Frontend: http://localhost:8501")
    print("=" * 60)
    print("🚀 Starting system...")
    print()

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'flask', 'flask_cors', 'requests', 
        'opencv-python', 'streamlit', 'pillow', 
        'numpy', 'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'pillow':
                import PIL
            elif package == 'flask_cors':
                import flask_cors
            else:
                __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing dependencies:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n📦 Install with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All dependencies installed!")
    return True

def run_api_server():
    """Run the API server in background"""
    try:
        print("🔧 Starting API Server...")
        process = subprocess.Popen([
            sys.executable, 'api_final.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for server to start
        time.sleep(3)
        
        # Check if server is running
        try:
            import requests
            response = requests.get('http://localhost:5000/health', timeout=5)
            if response.status_code == 200:
                print("✅ API Server running successfully!")
                return process
            else:
                print("❌ API Server failed to start")
                return None
        except:
            print("❌ API Server not responding")
            return None
            
    except Exception as e:
        print(f"❌ Failed to start API Server: {e}")
        return None

def run_streamlit_app():
    """Run Streamlit app"""
    try:
        print("🎨 Starting Streamlit App...")
        
        # Open browser after a delay
        def open_browser():
            time.sleep(5)
            webbrowser.open('http://localhost:8501')
        
        Thread(target=open_browser, daemon=True).start()
        
        # Run streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'app_final.py'
        ])
        
    except Exception as e:
        print(f"❌ Failed to start Streamlit App: {e}")

def main():
    """Main function"""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        input("\nPress Enter to exit...")
        return
    
    # Check if required files exist
    required_files = ['api_final.py', 'app_final.py']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ Missing required files:")
        for f in missing_files:
            print(f"   - {f}")
        print("\nMake sure these files are in the current directory.")
        input("\nPress Enter to exit...")
        return
    
    print("✅ All required files found!")
    
    # Start API server
    api_process = run_api_server()
    
    if api_process:
        try:
            print("\n🌟 System is ready!")
            print("📱 Open your browser and go to: http://localhost:8501")
            print("⚠️  Press Ctrl+C to stop the system")
            print()
            
            # Run Streamlit app (this will block)
            run_streamlit_app()
            
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping system...")
            
        finally:
            # Clean up API server
            if api_process:
                print("🔧 Stopping API Server...")
                api_process.terminate()
                api_process.wait()
                print("✅ API Server stopped")
    else:
        print("❌ Failed to start system")
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()