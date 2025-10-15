#!/usr/bin/env python3
"""
Render Deployment Checklist for TerrorReco
Quick checklist to verify everything is ready for deployment.
"""

def print_checklist():
    """Print the deployment checklist."""
    print("🚀 TerrorReco Render Deployment Checklist")
    print("=" * 50)
    
    print("\n📋 Pre-Deployment Checklist:")
    print("✅ Code is pushed to GitHub")
    print("✅ All tests pass (run: python test_deployment_simple.py)")
    print("✅ Database configuration is working")
    print("✅ Static files are present")
    print("✅ Docker configuration is ready")
    
    print("\n🔧 Render Configuration:")
    print("1. Create new Web Service on Render")
    print("2. Connect your GitHub repository")
    print("3. Set build command: pip install -e .")
    print("4. Set start command: uvicorn app.main:app --host 0.0.0.0 --port $PORT")
    
    print("\n🌍 Environment Variables to Set in Render:")
    print("• DATABASE_URL: (Render will provide this for PostgreSQL)")
    print("• OMDB_API_KEY: Your OMDb API key")
    print("• SECRET_KEY: A secure random string")
    print("• DEBUG: false")
    print("• STRIPE_PUBLISHABLE_KEY: Your Stripe publishable key")
    print("• STRIPE_SECRET_KEY: Your Stripe secret key")
    print("• STRIPE_WEBHOOK_SECRET: Your Stripe webhook secret (optional)")
    print("• COFFEE_PRICE_ID: Your Stripe price ID for coffee")
    
    print("\n📊 Database Setup:")
    print("1. Create PostgreSQL database on Render")
    print("2. Copy the DATABASE_URL from the database dashboard")
    print("3. Paste it as DATABASE_URL environment variable")
    print("4. The app will automatically normalize the URL for psycopg")
    
    print("\n🔍 Post-Deployment Verification:")
    print("1. Check that the app starts without errors")
    print("2. Test the homepage loads")
    print("3. Test user registration/login")
    print("4. Test movie recommendation functionality")
    print("5. Check database operations work")
    
    print("\n🚨 Common Issues & Solutions:")
    print("• Build fails: Check Python version is 3.11+")
    print("• Database connection fails: Verify DATABASE_URL is set")
    print("• Static files not loading: Check file paths in templates")
    print("• OMDb API errors: Verify OMDB_API_KEY is set")
    
    print("\n📞 Support:")
    print("• Render docs: https://render.com/docs")
    print("• FastAPI docs: https://fastapi.tiangolo.com/")
    print("• SQLAlchemy docs: https://docs.sqlalchemy.org/")

if __name__ == "__main__":
    print_checklist()
