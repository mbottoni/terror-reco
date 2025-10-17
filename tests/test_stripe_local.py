#!/usr/bin/env python3
"""
Local Stripe Integration Test Script for TerrorReco
Tests the complete coffee payment flow locally.
"""

import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


def print_header(title):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"🧪 {title}")
    print(f"{'='*60}")

def print_success(message):
    """Print a success message."""
    print(f"✅ {message}")

def print_error(message):
    """Print an error message."""
    print(f"❌ {message}")

def print_warning(message):
    """Print a warning message."""
    print(f"⚠️  {message}")

def print_info(message):
    """Print an info message."""
    print(f"ℹ️  {message}")

def check_environment_variables():
    """Check if Stripe environment variables are set."""
    print_header("Checking Environment Variables")
    
    required_vars = [
        'STRIPE_PUBLISHABLE_KEY',
        'STRIPE_SECRET_KEY', 
        'COFFEE_PRICE_ID'
    ]
    
    optional_vars = [
        'STRIPE_WEBHOOK_SECRET'
    ]
    
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Mask the secret key for display
            if 'SECRET' in var:
                display_value = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
            else:
                display_value = value
            print_success(f"{var}: {display_value}")
        else:
            print_error(f"{var}: Not set")
            missing_vars.append(var)
    
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print_success(f"{var}: {value[:8]}...{value[-4:]}")
        else:
            print_warning(f"{var}: Not set (optional)")
    
    if missing_vars:
        print_error(f"Missing required variables: {missing_vars}")
        print_info("Please set these in your .env file or environment")
        return False
    
    return True

def check_stripe_keys():
    """Validate Stripe key formats."""
    print_header("Validating Stripe Key Formats")
    
    publishable_key = os.getenv('STRIPE_PUBLISHABLE_KEY')
    secret_key = os.getenv('STRIPE_SECRET_KEY')
    
    if publishable_key:
        if publishable_key.startswith('pk_test_'):
            print_success("Publishable key is in test mode")
        elif publishable_key.startswith('pk_live_'):
            print_warning("Publishable key is in LIVE mode - be careful!")
        else:
            print_error("Publishable key format is invalid")
            return False
    
    if secret_key:
        if secret_key.startswith('sk_test_'):
            print_success("Secret key is in test mode")
        elif secret_key.startswith('sk_live_'):
            print_warning("Secret key is in LIVE mode - be careful!")
        else:
            print_error("Secret key format is invalid")
            return False
    
    return True

def test_stripe_connection():
    """Test connection to Stripe API."""
    print_header("Testing Stripe API Connection")
    
    try:
        import stripe
        stripe.api_key = os.getenv('STRIPE_SECRET_KEY')
        
        # Test API connection by retrieving account info
        account = stripe.Account.retrieve()
        print_success(f"Connected to Stripe account: {account.id}")
        print_success(f"Account type: {account.type}")
        
        # Test price retrieval
        price_id = os.getenv('COFFEE_PRICE_ID')
        if price_id:
            try:
                price = stripe.Price.retrieve(price_id)
                print_success(f"Price found: ${price.unit_amount/100:.2f} {price.currency.upper()}")
                print_success(f"Product: {price.product}")
            except stripe.error.InvalidRequestError as e:
                print_error(f"Price ID invalid: {e}")
                return False
        
        return True
        
    except ImportError:
        print_error("Stripe package not installed. Run: pip install stripe")
        return False
    except stripe.error.AuthenticationError:
        print_error("Stripe API key is invalid")
        return False
    except Exception as e:
        print_error(f"Stripe connection failed: {e}")
        return False

def start_local_server():
    """Start the local development server."""
    print_header("Starting Local Server")
    
    try:
        # Check if uvicorn is available
        result = subprocess.run([sys.executable, "-c", "import uvicorn; print('uvicorn available')"], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print_error("uvicorn not available. Install with: pip install uvicorn")
            return None
        
        print_info("Starting server on http://localhost:8000")
        print_info("Press Ctrl+C to stop the server")
        
        # Start the server
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "app.main:app", 
            "--reload", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ])
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Check if server is running
        try:
            import requests
            response = requests.get("http://localhost:8000", timeout=5)
            if response.status_code == 200:
                print_success("Server started successfully!")
                return process
        except ImportError:
            print_warning("requests not available, skipping server check")
            print_success("Server should be running on http://localhost:8000")
            return process
        except Exception:
            print_warning("Could not verify server status, but process started")
            return process
            
    except Exception as e:
        print_error(f"Failed to start server: {e}")
        return None

def test_coffee_button():
    """Test the coffee button functionality."""
    print_header("Testing Coffee Button")
    
    print_info("1. Open http://localhost:8000 in your browser")
    print_info("2. Scroll to the bottom of the page")
    print_info("3. Look for the '☕ Buy me a coffee' button")
    print_info("4. Click the button")
    
    try:
        webbrowser.open("http://localhost:8000")
        print_success("Opened browser to homepage")
    except Exception:
        print_warning("Could not open browser automatically")
        print_info("Please manually open http://localhost:8000")
    
    input("\nPress Enter after you've verified the coffee button is visible...")

def test_payment_flow():
    """Test the complete payment flow."""
    print_header("Testing Payment Flow")
    
    print_info("Now let's test the complete payment flow:")
    print_info("1. Click the '☕ Buy me a coffee' button")
    print_info("2. You should be redirected to /stripe/coffee")
    print_info("3. Click '☕ Buy me a coffee ($3)' button")
    print_info("4. You should be redirected to Stripe checkout")
    
    print_warning("IMPORTANT: Use test card numbers:")
    print_info("Card: 4242 4242 4242 4242")
    print_info("Expiry: Any future date (e.g., 12/25)")
    print_info("CVC: Any 3 digits (e.g., 123)")
    print_info("ZIP: Any 5 digits (e.g., 12345)")
    
    input("\nPress Enter after you've completed the payment test...")

def test_webhook_setup():
    """Test webhook setup with ngrok."""
    print_header("Testing Webhook Setup")
    
    print_info("For webhook testing, you need to expose your local server:")
    print_info("1. Install ngrok: https://ngrok.com/download")
    print_info("2. In a new terminal, run: ngrok http 8000")
    print_info("3. Copy the HTTPS URL (e.g., https://abc123.ngrok.io)")
    print_info("4. In Stripe Dashboard, create webhook endpoint:")
    print_info("   URL: https://abc123.ngrok.io/stripe/webhook")
    print_info("   Events: checkout.session.completed")
    print_info("5. Copy the webhook secret and add to environment")
    
    webhook_secret = os.getenv('STRIPE_WEBHOOK_SECRET')
    if webhook_secret:
        print_success("Webhook secret is configured")
    else:
        print_warning("Webhook secret not configured (optional for basic testing)")

def cleanup_and_summary():
    """Clean up and provide summary."""
    print_header("Test Summary")
    
    print_success("Local testing completed!")
    print_info("What you should have verified:")
    print("✅ Coffee button appears on homepage")
    print("✅ Payment page loads correctly")
    print("✅ Stripe checkout opens")
    print("✅ Test payment processes successfully")
    print("✅ Success page shows after payment")
    
    print_info("\nNext steps:")
    print("1. Deploy to Render")
    print("2. Set up production webhook")
    print("3. Test with real domain")
    print("4. Switch to live Stripe keys when ready")

def main():
    """Run the complete local test suite."""
    print("🧪 TerrorReco Stripe Integration - Local Testing")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("app/main.py").exists():
        print_error("Please run this script from the project root directory")
        return 1
    
    # Load environment variables
    env_file = Path(".env")
    if env_file.exists():
        print_info("Loading environment variables from .env file")
        from dotenv import load_dotenv
        load_dotenv()
    else:
        print_warning(".env file not found, using system environment variables")
    
    # Run tests
    tests = [
        ("Environment Variables", check_environment_variables),
        ("Stripe Key Formats", check_stripe_keys),
        ("Stripe API Connection", test_stripe_connection),
    ]
    
    failed_tests = []
    
    for test_name, test_func in tests:
        try:
            if not test_func():
                failed_tests.append(test_name)
        except Exception as e:
            print_error(f"{test_name} test crashed: {e}")
            failed_tests.append(test_name)
    
    if failed_tests:
        print_error(f"Tests failed: {failed_tests}")
        print_info("Please fix the issues before proceeding")
        return 1
    
    # Start server and test
    print_header("Starting Interactive Testing")
    
    server_process = start_local_server()
    if not server_process:
        return 1
    
    try:
        test_coffee_button()
        test_payment_flow()
        test_webhook_setup()
        cleanup_and_summary()
        
    except KeyboardInterrupt:
        print_info("\nTesting interrupted by user")
    finally:
        if server_process:
            print_info("Stopping server...")
            server_process.terminate()
            server_process.wait()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
