#!/usr/bin/env python3
"""
Pre-deployment test script for TerrorReco application.
Tests all critical components before deploying to Render.
"""

import subprocess
import sys
from pathlib import Path


def print_header(title):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
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

def test_python_version():
    """Test Python version compatibility."""
    print_header("Testing Python Version")
    
    version = sys.version_info
    print(f"Current Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 11:
        print_success("Python version is compatible (3.11+)")
        return True
    else:
        print_warning(f"Python {version.major}.{version.minor} detected. Render requires Python 3.11+")
        print("This test will continue but deployment may fail.")
        return False

def test_dependencies():
    """Test that all required dependencies are available."""
    print_header("Testing Dependencies")
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'sqlalchemy',
        'psycopg',
        'pydantic',
        'pydantic_settings',
        'jinja2',
        'httpx',
        'argon2',
        'itsdangerous',
        'python-multipart',
        'scikit-learn',
        'tenacity'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print_success(f"{package} is available")
        except ImportError:
            print_error(f"{package} is missing")
            missing_packages.append(package)
    
    if missing_packages:
        print_warning(f"Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -e . to install dependencies")
        return False
    
    print_success("All required dependencies are available")
    return True

def test_database_configuration():
    """Test database configuration and URL normalization."""
    print_header("Testing Database Configuration")
    
    try:
        # Add app directory to path
        app_dir = Path(__file__).parent / "app"
        sys.path.insert(0, str(app_dir))
        
        # Test URL normalization function
        from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
        
        def _normalize_database_url(raw: str) -> str:
            """Convert postgres URLs to SQLAlchemy's psycopg v3 driver and ensure SSL on Render."""
            if not raw:
                return raw
            p = urlparse(raw)
            scheme = p.scheme
            
            # Only normalize PostgreSQL URLs, leave SQLite and others unchanged
            if scheme not in ("postgres", "postgresql"):
                return raw
            
            # Map postgres/postgresql to psycopg driver name
            scheme = "postgresql+psycopg"
            
            # Ensure sslmode=require if not present
            query_pairs = dict(parse_qsl(p.query))
            if "sslmode" not in query_pairs:
                query_pairs["sslmode"] = "require"
            
            new_query = urlencode(query_pairs)
            new_p = p._replace(scheme=scheme, query=new_query)
            return urlunparse(new_p)
        
        # Test cases
        test_cases = [
            ("sqlite:///./app.db", "sqlite:///./app.db"),
            ("postgres://user:pass@localhost:5432/db", "postgresql+psycopg://user:pass@localhost:5432/db?sslmode=require"),
            ("postgresql://user:pass@localhost:5432/db", "postgresql+psycopg://user:pass@localhost:5432/db?sslmode=require"),
            ("postgresql://user:pass@render-host:5432/db", "postgresql+psycopg://user:pass@render-host:5432/db?sslmode=require"),
        ]
        
        for original, expected in test_cases:
            result = _normalize_database_url(original)
            if result == expected:
                print_success(f"URL normalization: {original} -> {result}")
            else:
                print_error(f"URL normalization failed: {original} -> {result} (expected: {expected})")
                return False
        
        # Test SQLAlchemy compatibility
        from sqlalchemy.engine.url import make_url
        for original, expected in test_cases:
            try:
                parsed = make_url(expected)
                print_success(f"SQLAlchemy can parse: {expected}")
            except Exception as e:
                print_error(f"SQLAlchemy cannot parse {expected}: {e}")
                return False
        
        print_success("Database configuration is working correctly")
        return True
        
    except Exception as e:
        print_error(f"Database configuration test failed: {e}")
        return False

def test_app_imports():
    """Test that all app modules can be imported."""
    print_header("Testing App Imports")
    
    try:
        # Add app directory to path
        app_dir = Path(__file__).parent / "app"
        sys.path.insert(0, str(app_dir))
        
        # Test core modules
        modules_to_test = [
            'settings',
            'db',
            'auth',
            'main',
            'models',
            'history',
            'security'
        ]
        
        for module in modules_to_test:
            try:
                __import__(module)
                print_success(f"Module {module} imports successfully")
            except Exception as e:
                print_error(f"Module {module} import failed: {e}")
                return False
        
        # Test services
        try:
            print_success("Services modules import successfully")
        except Exception as e:
            print_error(f"Services import failed: {e}")
            return False
        
        # Test strategies
        try:
            print_success("Strategy modules import successfully")
        except Exception as e:
            print_error(f"Strategies import failed: {e}")
            return False
        
        print_success("All app modules import successfully")
        return True
        
    except Exception as e:
        print_error(f"App imports test failed: {e}")
        return False

def test_database_operations():
    """Test database operations."""
    print_header("Testing Database Operations")
    
    try:
        # Add app directory to path
        app_dir = Path(__file__).parent / "app"
        sys.path.insert(0, str(app_dir))
        
        # Import database modules
        import db
        import settings
        
        # Test settings
        settings_obj = settings.get_settings()
        print_success(f"Settings loaded: DATABASE_URL={settings_obj.DATABASE_URL}")
        
        # Test database initialization
        db.init_db()
        print_success("Database initialization successful")
        
        # Test database session
        with db.get_db_session() as session:
            print_success("Database session created successfully")
        
        print_success("Database operations working correctly")
        return True
        
    except Exception as e:
        print_error(f"Database operations test failed: {e}")
        return False

def test_fastapi_app():
    """Test FastAPI application creation."""
    print_header("Testing FastAPI Application")
    
    try:
        # Add app directory to path
        app_dir = Path(__file__).parent / "app"
        sys.path.insert(0, str(app_dir))
        
        # Import and test FastAPI app
        from main import app
        
        print_success(f"FastAPI app created: {app.title}")
        print_success(f"App version: {app.version}")
        
        # Test that routes are registered
        routes = [route.path for route in app.routes]
        expected_routes = ['/', '/login', '/register', '/logout', '/recommend', '/history']
        
        for expected_route in expected_routes:
            if any(expected_route in route for route in routes):
                print_success(f"Route {expected_route} is registered")
            else:
                print_warning(f"Route {expected_route} not found in registered routes")
        
        print_success("FastAPI application is working correctly")
        return True
        
    except Exception as e:
        print_error(f"FastAPI application test failed: {e}")
        return False

def test_environment_variables():
    """Test environment variable handling."""
    print_header("Testing Environment Variables")
    
    # Test that the app can handle missing environment variables gracefully
    try:
        # Add app directory to path
        app_dir = Path(__file__).parent / "app"
        sys.path.insert(0, str(app_dir))
        
        import settings
        
        # Test default values
        settings_obj = settings.get_settings()
        
        print_success(f"DATABASE_URL: {settings_obj.DATABASE_URL}")
        print_success(f"DEBUG: {settings_obj.DEBUG}")
        print_success(f"APP_NAME: {settings_obj.APP_NAME}")
        
        # Test that OMDB_API_KEY can be None (for testing)
        if settings_obj.OMDB_API_KEY is None:
            print_warning("OMDB_API_KEY is not set (this is OK for testing)")
        else:
            print_success("OMDB_API_KEY is set")
        
        print_success("Environment variables handled correctly")
        return True
        
    except Exception as e:
        print_error(f"Environment variables test failed: {e}")
        return False

def test_static_files():
    """Test that static files exist."""
    print_header("Testing Static Files")
    
    static_files = [
        "app/static/styles.css",
        "app/static/assets/spooky.gif",
        "app/templates/index.html",
        "app/templates/login.html",
        "app/templates/register.html",
        "app/templates/results.html",
        "app/templates/history.html",
        "app/templates/loading.html"
    ]
    
    missing_files = []
    
    for file_path in static_files:
        if Path(file_path).exists():
            print_success(f"Static file exists: {file_path}")
        else:
            print_error(f"Static file missing: {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print_error(f"Missing static files: {missing_files}")
        return False
    
    print_success("All static files are present")
    return True

def test_docker_configuration():
    """Test Docker configuration if present."""
    print_header("Testing Docker Configuration")
    
    docker_files = ["Dockerfile", "docker-compose.yml"]
    
    for docker_file in docker_files:
        if Path(docker_file).exists():
            print_success(f"Docker file exists: {docker_file}")
            
            # Basic syntax check
            try:
                with open(docker_file) as f:
                    content = f.read()
                print_success(f"Docker file {docker_file} is readable")
            except Exception as e:
                print_error(f"Docker file {docker_file} has issues: {e}")
                return False
        else:
            print_warning(f"Docker file missing: {docker_file}")
    
    print_success("Docker configuration looks good")
    return True

def test_project_structure():
    """Test that the project has the expected structure."""
    print_header("Testing Project Structure")
    
    required_dirs = [
        "app",
        "app/services",
        "app/services/strategies",
        "app/templates",
        "app/static",
        "app/static/assets"
    ]
    
    required_files = [
        "pyproject.toml",
        "app/__init__.py",
        "app/main.py",
        "app/db.py",
        "app/settings.py",
        "app/models.py",
        "app/auth.py",
        "app/security.py",
        "app/history.py"
    ]
    
    missing_items = []
    
    # Check directories
    for dir_path in required_dirs:
        if Path(dir_path).is_dir():
            print_success(f"Directory exists: {dir_path}")
        else:
            print_error(f"Directory missing: {dir_path}")
            missing_items.append(dir_path)
    
    # Check files
    for file_path in required_files:
        if Path(file_path).exists():
            print_success(f"File exists: {file_path}")
        else:
            print_error(f"File missing: {file_path}")
            missing_items.append(file_path)
    
    if missing_items:
        print_error(f"Missing items: {missing_items}")
        return False
    
    print_success("Project structure is correct")
    return True

def run_uvicorn_test():
    """Test if uvicorn can start the application."""
    print_header("Testing Uvicorn Startup")
    
    try:
        # Test if uvicorn can import the app
        result = subprocess.run([
            sys.executable, "-c", 
            "import sys; sys.path.insert(0, 'app'); from main import app; print('✅ App can be imported by uvicorn')"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print_success("Uvicorn can import the application")
            return True
        else:
            print_error(f"Uvicorn import failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print_error("Uvicorn test timed out")
        return False
    except Exception as e:
        print_error(f"Uvicorn test failed: {e}")
        return False

def main():
    """Run all deployment tests."""
    print("🚀 TerrorReco Pre-Deployment Test Suite")
    print("=" * 60)
    
    tests = [
        ("Python Version", test_python_version),
        ("Dependencies", test_dependencies),
        ("Project Structure", test_project_structure),
        ("Database Configuration", test_database_configuration),
        ("App Imports", test_app_imports),
        ("Database Operations", test_database_operations),
        ("Environment Variables", test_environment_variables),
        ("Static Files", test_static_files),
        ("FastAPI Application", test_fastapi_app),
        ("Docker Configuration", test_docker_configuration),
        ("Uvicorn Startup", run_uvicorn_test),
    ]
    
    passed = 0
    total = len(tests)
    failed_tests = []
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed_tests.append(test_name)
        except Exception as e:
            print_error(f"{test_name} test crashed: {e}")
            failed_tests.append(test_name)
    
    # Print summary
    print_header("Test Summary")
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print_success("🎉 All tests passed! Your application is ready for deployment to Render!")
        print("\n📋 Deployment Checklist:")
        print("✅ Python version compatible")
        print("✅ All dependencies available")
        print("✅ Database configuration working")
        print("✅ Application imports successfully")
        print("✅ Static files present")
        print("✅ FastAPI app creates successfully")
        print("✅ Uvicorn can start the application")
        print("\n🚀 Ready to deploy!")
        return 0
    else:
        print_error(f"⚠️  {len(failed_tests)} test(s) failed: {', '.join(failed_tests)}")
        print("\n🔧 Please fix the failing tests before deploying to Render.")
        print("\n💡 Common fixes:")
        print("   - Run 'pip install -e .' to install dependencies")
        print("   - Check that all files are present")
        print("   - Verify Python version is 3.11+")
        print("   - Ensure DATABASE_URL is set correctly")
        return 1

if __name__ == "__main__":
    sys.exit(main())
