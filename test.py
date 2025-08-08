#!/usr/bin/env python3
"""
Local testing script for the Data Analyst Agent
Run this to test your API locally before deploying
"""

import requests
import json
import os
from io import StringIO

def test_health_check():
    """Test if server is running"""
    try:
        response = requests.get('http://127.0.0.1:5000/')
        print("✅ Health check passed")
        print(f"Response: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_api_get():
    """Test API GET request"""
    try:
        response = requests.get('http://127.0.0.1:5000/api/')
        print("✅ API GET test passed")
        print(f"Response: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ API GET test failed: {e}")
        return False

def test_simple_question():
    """Test with a simple question"""
    try:
        # Create test question
        questions_text = """
1. What is 2 + 2?
2. What is the capital of France?
        """.strip()
        
        files = {
            'questions.txt': ('questions.txt', questions_text, 'text/plain')
        }
        
        response = requests.post('http://127.0.0.1:5000/api/', files=files)
        print("✅ Simple question test passed")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Simple question test failed: {e}")
        print(f"Response text: {response.text if 'response' in locals() else 'No response'}")
        return False

def test_wikipedia_scraping():
    """Test Wikipedia scraping functionality"""
    try:
        questions_text = """
Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array of strings containing the answer.

1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
   Return as a base-64 encoded data URI, `"data:image/png;base64,iVBORw0KG..."` under 100,000 bytes.
        """.strip()
        
        files = {
            'questions.txt': ('questions.txt', questions_text, 'text/plain')
        }
        
        print("🔄 Testing Wikipedia scraping (this may take 30-60 seconds)...")
        response = requests.post('http://127.0.0.1:5000/api/', files=files, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Wikipedia scraping test passed")
            print(f"Result type: {type(result)}")
            if isinstance(result, list):
                print(f"Array length: {len(result)}")
                for i, item in enumerate(result):
                    if isinstance(item, str) and item.startswith('data:image'):
                        print(f"Item {i}: Image data ({len(item)} chars)")
                    else:
                        print(f"Item {i}: {item}")
            else:
                print(f"Result: {result}")
            return True
        else:
            print(f"❌ Wikipedia scraping test failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Wikipedia scraping test failed: {e}")
        return False

def create_test_files():
    """Create test files for debugging"""
    
    # Simple questions file
    with open('test_simple.txt', 'w') as f:
        f.write("1. What is 2 + 2?\n2. What is the capital of France?")
    
    # Wikipedia questions file
    with open('test_wikipedia.txt', 'w') as f:
        f.write("""Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array of strings containing the answer.

1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
   Return as a base-64 encoded data URI, `"data:image/png;base64,iVBORw0KG..."` under 100,000 bytes.""")
    
    print("📄 Created test files: test_simple.txt, test_wikipedia.txt")

def main():
    print("🚀 Starting Data Analyst Agent Tests")
    print("="*50)
    
    # Create test files
    create_test_files()
    
    # Run tests
    tests = [
        ("Health Check", test_health_check),
        ("API GET", test_api_get),
        ("Simple Question", test_simple_question),
        ("Wikipedia Scraping", test_wikipedia_scraping)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        success = test_func()
        results.append((test_name, success))
        print("-" * 30)
    
    # Summary
    print("\n📊 Test Summary:")
    print("=" * 50)
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your API is ready for deployment.")
    else:
        print("⚠️  Some tests failed. Check the errors above and fix before deploying.")

if __name__ == "__main__":
    main()