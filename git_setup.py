#!/usr/bin/env python3
"""
Git Setup Script for Medical Image Classifier

Initializes Git repository and prepares for first commit.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(f"   Error: {e.stderr.strip()}")
        return False

def main():
    """Main setup function"""
    print("🚀 Medical Image Classifier - Git Setup")
    print("=" * 50)
    
    # Check if git is installed
    if not run_command("git --version", "Checking Git installation"):
        print("❌ Git is not installed. Please install Git first.")
        return False
    
    # Initialize git repository
    if not os.path.exists('.git'):
        if not run_command("git init", "Initializing Git repository"):
            return False
    else:
        print("✅ Git repository already exists")
    
    # Add all files
    if not run_command("git add .", "Adding all files to Git"):
        return False
    
    # Create initial commit
    commit_message = "Initial commit: Medical Image Classifier v1.0.0"
    if not run_command(f'git commit -m "{commit_message}"', "Creating initial commit"):
        print("ℹ️  Note: If this failed, you might need to configure Git:")
        print("   git config --global user.name 'Your Name'")
        print("   git config --global user.email 'your.email@example.com'")
        return False
    
    print("\n🎉 Git setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Create a repository on GitHub")
    print("2. Add remote origin:")
    print("   git remote add origin https://github.com/yourusername/medical-image-classifier.git")
    print("3. Push to GitHub:")
    print("   git branch -M main")
    print("   git push -u origin main")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)