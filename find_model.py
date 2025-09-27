import os
import glob

print("🔍 Searching for model files...")

# Search for all .pth files
model_files = glob.glob("**/*.pth", recursive=True) + glob.glob("**/*.pt", recursive=True)

if model_files:
    print("✅ Found model files:")
    for file in model_files:
        size = os.path.getsize(file) / (1024*1024)  # Size in MB
        print(f"   📁 {file} ({size:.1f} MB)")
else:
    print("❌ No model files found!")
    print("💡 Make sure training completed successfully")