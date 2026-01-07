"""
Run the Biometric Security System
"""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Create required directories
os.makedirs('data', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Run the app
from app import app, config

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🔐 Biometric Security System")
    print("="*50)
    print(f"Starting server at http://localhost:{config['app']['port']}")
    print("="*50 + "\n")
    
    app.run(
        host=config['app']['host'],
        port=config['app']['port'],
        debug=config['app']['debug']
    )
