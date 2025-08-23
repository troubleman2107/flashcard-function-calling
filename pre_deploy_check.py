#!/usr/bin/env python3
"""
Pre-deployment checker for Flashcard Application
"""

import sys
import os

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def check_dependencies():
    """Test if all required modules can be imported"""
    print("\n🧪 Checking dependencies...")
    
    modules_to_test = [
        "flask",
        "transformers", 
        "openai",
        "torch",
        "soundfile",
        "dotenv",
        "chromadb",
        "langchain",
        "langchain_openai",
        "pydantic",
    ]
    
    failed_modules = []
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✅ {module_name}")
        except ImportError:
            print(f"❌ {module_name} - NOT FOUND")
            failed_modules.append(module_name)
    
    if failed_modules:
        print(f"\n❌ {len(failed_modules)} modules missing. Run: pip install -r requirements.txt")
        return False
    else:
        print(f"\n✅ All {len(modules_to_test)} dependencies found!")
        return True

def check_environment():
    """Check environment configuration"""
    print("\n🔧 Checking environment...")
    
    # Check for .env file
    if os.path.exists('.env'):
        print("✅ .env file found")
        
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            if os.getenv('OPENAI_API_KEY'):
                # Hide the key for security
                key = os.getenv('OPENAI_API_KEY')
                masked_key = key[:8] + "..." + key[-4:] if len(key) > 12 else "***"
                print(f"✅ OPENAI_API_KEY found ({masked_key})")
                return True
            else:
                print("❌ OPENAI_API_KEY not found in .env")
                return False
        except Exception as e:
            print(f"❌ Error loading .env: {e}")
            return False
    else:
        print("❌ .env file not found")
        print("💡 Create .env with: OPENAI_API_KEY=your-key-here")
        return False

def check_files():
    """Check if required files exist"""
    print("\n📁 Checking required files...")
    
    required_files = [
        "app.py",
        "requirements.txt", 
        "templates/chat.html",
        "static/styles.css"
    ]
    
    all_exist = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_exist = False
    
    return all_exist

def main():
    print("🔍 Flashcard Application - Pre-Deployment Check")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version()),
        ("Dependencies", check_dependencies()),
        ("Environment", check_environment()),
        ("Required Files", check_files())
    ]
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY:")
    
    all_passed = True
    for name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("🎉 ALL CHECKS PASSED! Ready for deployment!")
        print("📖 See DEPLOYMENT_STEPS.md for deployment instructions")
    else:
        print("❌ Some checks failed. Fix the issues above before deploying.")
    
    return all_passed

if __name__ == "__main__":
    main()
