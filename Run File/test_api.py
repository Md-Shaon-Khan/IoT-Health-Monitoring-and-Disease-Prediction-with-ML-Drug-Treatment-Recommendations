# Create a separate test script for the API
test_api_code = '''
# test_api.py
"""
Test script for FastAPI Health Prediction System
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_api_endpoints():
    """Test all API endpoints"""
    print("Testing FastAPI endpoints...")
    print("="*60)
    
    # Test 1: Root endpoint
    print("\\n1. Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 2: Health check
    print("\\n2. Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 3: Model info
    print("\\n3. Testing model info...")
    try:
        response = requests.get(f"{BASE_URL}/model-info")
        print(f"   Status: {response.status_code}")
        data = response.json()
        print(f"   Model accuracy: {data['accuracy']:.2%}")
        print(f"   Features: {len(data['features'])}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 4: Make prediction (Normal case)
    print("\\n4. Testing prediction (Normal case)...")
    try:
        normal_data = {
            "temperature": 36.5,
            "heart_rate": 75.0,
            "bp_sys": 120.0,
            "bp_dia": 80.0,
            "humidity": 40.0,
            "fever": 0,
            "cough": 0,
            "chest_pain": 0,
            "shortness_of_breath": 0,
            "fatigue": 0,
            "headache": 0
        }
        
        response = requests.post(f"{BASE_URL}/predict", json=normal_data)
        print(f"   Status: {response.status_code}")
        result = response.json()
        print(f"   Prediction: {result['disease']}")
        print(f"   Confidence: {result['confidence']:.2%}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 5: Make prediction (Heart Risk case)
    print("\\n5. Testing prediction (Heart Risk case)...")
    try:
        heart_data = {
            "temperature": 37.0,
            "heart_rate": 105.0,
            "bp_sys": 150.0,
            "bp_dia": 95.0,
            "humidity": 40.0,
            "fever": 0,
            "cough": 0,
            "chest_pain": 1,
            "shortness_of_breath": 1,
            "fatigue": 1,
            "headache": 0
        }
        
        response = requests.post(f"{BASE_URL}/predict", json=heart_data)
        print(f"   Status: {response.status_code}")
        result = response.json()
        print(f"   Prediction: {result['disease']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   Actions: {result['suggested_actions'][:100]}...")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\\n" + "="*60)
    print("API testing complete!")

if __name__ == "__main__":
    print("HEALTH PREDICTION SYSTEM - API TEST")
    print("="*60)
    print("Make sure the FastAPI server is running on http://localhost:8000")
    print()
    test_api_endpoints()
'''

# Save the test script
with open('test_api.py', 'w') as f:
    f.write(test_api_code)

print("✓ FastAPI app created as 'app.py'")
print("✓ API test script created as 'test_api.py'")
print("\nTo run the FastAPI server:")
print("  python app.py")
print("\nTo test the API (after starting server):")
print("  python test_api.py")
print("\nAPI documentation will be available at: http://localhost:8000/docs")