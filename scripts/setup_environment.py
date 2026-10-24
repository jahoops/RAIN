"""
Script to install required packages for RAIN project models
"""
import subprocess
import sys

def install_requirements():
    """Install packages from requirements.txt"""
    print("Installing required packages...")
    
    try:
        # Use pip to install packages
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("All packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        print("\nYou might want to try installing the key packages individually:")
        print("pip install tensorflow ngboost")
    
    # Additional instructions for TensorFlow
    print("\nNote: If you encounter issues with TensorFlow:")
    print("- For CPU only: pip install tensorflow")
    print("- For NVIDIA GPU support: pip install tensorflow-gpu")
    print("- Make sure CUDA and cuDNN are installed for GPU support")
    
    # Check if packages can be imported
    print("\nVerifying installations:")
    packages_to_check = [
        "tensorflow", 
        "ngboost", 
        "xgboost", 
        "lightgbm",
        "catboost"
    ]
    
    for package in packages_to_check:
        try:
            __import__(package)
            print(f"✓ {package} is installed correctly")
        except ImportError:
            print(f"✕ {package} is not installed or has issues")

if __name__ == "__main__":
    install_requirements()
