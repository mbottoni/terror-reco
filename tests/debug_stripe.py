#!/usr/bin/env python3
"""
Debug script for Stripe integration issues on Render
"""

import os

from dotenv import load_dotenv


def check_render_environment():
    """Check if we're running on Render and what environment variables are available."""
    print("🔍 Environment Debug")
    print("=" * 40)
    
    # Check if running on Render
    if os.getenv('RENDER'):
        print("✅ Running on Render")
    else:
        print("❌ Not running on Render")
    
    # Check environment variables
    stripe_vars = [
        'STRIPE_PUBLISHABLE_KEY',
        'STRIPE_SECRET_KEY', 
        'COFFEE_PRICE_ID',
        'STRIPE_WEBHOOK_SECRET'
    ]
    
    print("\n📋 Stripe Environment Variables:")
    for var in stripe_vars:
        value = os.getenv(var)
        if value:
            if 'SECRET' in var:
                display_value = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
            else:
                display_value = value
            print(f"✅ {var}: {display_value}")
        else:
            print(f"❌ {var}: Not set")
    
    # Check other important vars
    other_vars = ['DATABASE_URL', 'SECRET_KEY', 'OMDB_API_KEY']
    print("\n📋 Other Environment Variables:")
    for var in other_vars:
        value = os.getenv(var)
        if value:
            if 'SECRET' in var or 'DATABASE' in var:
                display_value = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
            else:
                display_value = value
            print(f"✅ {var}: {display_value}")
        else:
            print(f"❌ {var}: Not set")

def test_stripe_connection():
    """Test Stripe connection with current environment."""
    print("\n🔗 Testing Stripe Connection")
    print("=" * 40)
    
    try:
        import stripe
        
        secret_key = os.getenv('STRIPE_SECRET_KEY')
        if not secret_key:
            print("❌ STRIPE_SECRET_KEY not set")
            return False
        
        stripe.api_key = secret_key
        
        # Test account connection
        account = stripe.Account.retrieve()
        print(f"✅ Connected to Stripe account: {account.id}")
        print(f"✅ Account type: {account.type}")
        
        # Test price
        price_id = os.getenv('COFFEE_PRICE_ID')
        if not price_id:
            print("❌ COFFEE_PRICE_ID not set")
            return False
        
        price = stripe.Price.retrieve(price_id)
        print(f"✅ Price found: ${price.unit_amount/100:.2f} {price.currency.upper()}")
        print(f"✅ Product: {price.product}")
        
        return True
        
    except ImportError:
        print("❌ Stripe package not installed")
        return False
    except Exception as e:
        print(f"❌ Stripe connection failed: {e}")
        return False

def test_checkout_session():
    """Test creating a checkout session."""
    print("\n🛒 Testing Checkout Session Creation")
    print("=" * 40)
    
    try:
        import stripe
        
        secret_key = os.getenv('STRIPE_SECRET_KEY')
        price_id = os.getenv('COFFEE_PRICE_ID')
        
        if not secret_key or not price_id:
            print("❌ Missing required environment variables")
            return False
        
        stripe.api_key = secret_key
        
        # Test creating a checkout session
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': price_id,
                'quantity': 1,
            }],
            mode='payment',
            success_url='https://example.com/success',
            cancel_url='https://example.com/cancel',
            metadata={
                'app': 'terror-reco',
                'type': 'coffee'
            }
        )
        
        print(f"✅ Checkout session created: {checkout_session.id}")
        print(f"✅ Checkout URL: {checkout_session.url}")
        
        return True
        
    except Exception as e:
        print(f"❌ Checkout session creation failed: {e}")
        return False

def main():
    """Run all debug checks."""
    print("🐛 TerrorReco Stripe Debug")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    checks = [
        ("Environment Check", check_render_environment),
        ("Stripe Connection", test_stripe_connection),
        ("Checkout Session", test_checkout_session),
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        try:
            if check_func():
                passed += 1
        except Exception as e:
            print(f"❌ {check_name} crashed: {e}")
    
    print(f"\n📊 Debug Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("✅ All checks passed! Stripe should work.")
    else:
        print("❌ Some checks failed. Check the issues above.")
        print("\n🔧 Common fixes:")
        print("1. Set environment variables in Render dashboard")
        print("2. Check Stripe keys are correct")
        print("3. Verify price ID exists")
        print("4. Redeploy after fixing environment variables")

if __name__ == "__main__":
    main()
