#!/usr/bin/env python3
"""
Simple startup script that runs holiday data scripts first, 
then the weather imputation script.
"""

import os
import sys
import subprocess

# Get the current directory where this script is located
main_dir = os.path.dirname(os.path.abspath(__file__))

# Define the paths to the scripts
bank_holidays_script = os.path.join(main_dir, "data", "HolidayData", "create_bank_holidays.py")
school_holidays_script = os.path.join(main_dir, "data", "HolidayData", "create_school_holidays.py")
weather_imputation_script = os.path.join(main_dir, "data", "WeatherImputation", "Wetter_Imputation.py")

# Run the holiday data scripts first
print("Running bank holidays script...")
subprocess.run([sys.executable, bank_holidays_script])

print("Running school holidays script...")
subprocess.run([sys.executable, school_holidays_script])

# Then run the weather imputation script
print("Running weather imputation script...")
subprocess.run([sys.executable, weather_imputation_script])

print("All done!")