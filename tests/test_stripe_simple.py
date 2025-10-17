#!/usr/bin/env python3
"""
Simple Stripe Integration Test
Tests the coffee button and payment flow without starting the full server.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def test_stripe_configuration():
    """Test Stripe configuration."""
    print("🧪 Testing Stripe Configuration")
    print("=" * 40)
    
    # Load environment variables
    load_dotenv()
    
    # Check environment variables
    required_vars = {
        'STRIPE_PUBLISHABLE_KEY': os.getenv('STRIPE_PUBLISHABLE_KEY'),
        'STRIPE_SECRET_KEY': os.getenv('STRIPE_SECRET_KEY'),
        'COFFEE_PRICE_ID': os.getenv('COFFEE_PRICE_ID'),
    }
    
    print("📋 Environment Variables:")
    for var, value in required_vars.items():
        if value:
            if 'SECRET' in var:
                display_value = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
            else:
                display_value = value
            print(f"✅ {var}: {display_value}")
        else:
            print(f"❌ {var}: Not set")
            return False
    
    # Test Stripe API connection
    try:
        import stripe
        stripe.api_key = required_vars['STRIPE_SECRET_KEY']
        
        print("\n🔗 Testing Stripe API Connection:")
        account = stripe.Account.retrieve()
        print(f"✅ Connected to account: {account.id}")
        print(f"✅ Account type: {account.type}")
        
        # Test price
        price = stripe.Price.retrieve(required_vars['COFFEE_PRICE_ID'])
        print(f"✅ Price: ${price.unit_amount/100:.2f} {price.currency.upper()}")
        print(f"✅ Product: {price.product}")
        
        return True
        
    except ImportError:
        print("❌ Stripe package not installed")
        return False
    except Exception as e:
        print(f"❌ Stripe connection failed: {e}")
        return False

def test_coffee_button_html():
    """Test that the coffee button HTML is correct."""
    print("\n🎨 Testing Coffee Button HTML")
    print("=" * 40)
    
    index_file = Path("app/templates/index.html")
    if not index_file.exists():
        print("❌ index.html not found")
        return False
    
    content = index_file.read_text()
    
    # Check for coffee button
    if "Buy me a coffee" in content:
        print("✅ Coffee button text found")
    else:
        print("❌ Coffee button text not found")
        return False
    
    if "☕" in content:
        print("✅ Coffee emoji found")
    else:
        print("❌ Coffee emoji not found")
        return False
    
    if "/stripe/coffee" in content:
        print("✅ Coffee button link found")
    else:
        print("❌ Coffee button link not found")
        return False
    
    return True

def test_stripe_templates():
    """Test that Stripe templates exist."""
    print("\n📄 Testing Stripe Templates")
    print("=" * 40)
    
    templates = [
        "app/templates/coffee.html",
        "app/templates/coffee_success.html", 
        "app/templates/coffee_cancel.html"
    ]
    
    for template in templates:
        if Path(template).exists():
            print(f"✅ {template} exists")
        else:
            print(f"❌ {template} missing")
            return False
    
    return True

def test_stripe_module():
    """Test that Stripe module can be imported."""
    print("\n🐍 Testing Stripe Module")
    print("=" * 40)
    
    try:
        sys.path.insert(0, "app")
        import stripe_payments
        print("✅ Stripe payments module imports successfully")
        
        # Check if router exists
        if hasattr(stripe_payments, 'router'):
            print("✅ Router exists")
        else:
            print("❌ Router not found")
            return False
            
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Module error: {e}")
        return False

def show_testing_instructions():
    """Show instructions for manual testing."""
    print("\n🚀 Manual Testing Instructions")
    print("=" * 40)
    
    print("To test the complete flow:")
    print("1. Install FastAPI dependencies:")
    print("   pip install fastapi uvicorn")
    print()
    print("2. Start the server:")
    print("   uvicorn app.main:app --reload")
    print()
    print("3. Open browser to: http://localhost:8000")
    print("4. Scroll to bottom and click '☕ Buy me a coffee'")
    print("5. Test with Stripe test card: 4242 4242 4242 4242")
    print()
    print("⚠️  WARNING: You're using LIVE Stripe keys!")
    print("   Consider switching to test keys for development:")
    print("   STRIPE_PUBLISHABLE_KEY=pk_test_...")
    print("   STRIPE_SECRET_KEY=sk_test_...")

def main():
    """Run all tests."""
    print("🧪 TerrorReco Stripe Integration Test")
    print("=" * 50)
    
    tests = [
        ("Stripe Configuration", test_stripe_configuration),
        ("Coffee Button HTML", test_coffee_button_html),
        ("Stripe Templates", test_stripe_templates),
        ("Stripe Module", test_stripe_module),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"\n❌ {test_name} test failed")
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Stripe integration is ready.")
        show_testing_instructions()
        return 0
    else:
        print("⚠️  Some tests failed. Please fix the issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
