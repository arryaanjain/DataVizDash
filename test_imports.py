"""
Test file to verify module imports are working correctly.
This is especially useful for debugging deployment issues.
"""
import sys
import os

# Print Python version and path information
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")
print(f"sys.path: {sys.path}")

# Try importing the problematic module
try:
    from analytics.growth_analysis import show_growth_analysis
    print("✅ Successfully imported show_growth_analysis from analytics.growth_analysis")
except ImportError as e:
    print(f"❌ Error importing show_growth_analysis: {str(e)}")
    
    # Try to get more information about the analytics module
    try:
        import analytics
        print(f"analytics module exists at: {analytics.__file__}")
        print(f"analytics module contains: {dir(analytics)}")
    except ImportError as e:
        print(f"❌ Error importing analytics module: {str(e)}")

# Print a list of all files in the analytics directory
try:
    analytics_dir = os.path.join(os.getcwd(), "analytics")
    print(f"Contents of analytics directory ({analytics_dir}):")
    for item in os.listdir(analytics_dir):
        item_path = os.path.join(analytics_dir, item)
        if os.path.isfile(item_path):
            print(f"  - File: {item} ({os.path.getsize(item_path)} bytes)")
        else:
            print(f"  - Directory: {item}")
except Exception as e:
    print(f"❌ Error listing analytics directory: {str(e)}")
