import pandas as pd
from datetime import datetime, timedelta

def expand_date_range(start_date, end_date):
    """Generate all dates between start and end date inclusive"""
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)
    return dates

# Define the holiday periods with their date ranges
holiday_periods = [
    # 2013
    ("Weihnachten", "01.01.2013", "05.01.2013"),
    ("Ostern", "25.03.2013", "09.04.2013"),
    ("Pfingsten", "10.05.2013", "10.05.2013"),
    ("Sommer", "24.06.2013", "03.08.2013"),
    ("Herbst", "04.10.2013", "18.10.2013"),
    ("Weihnachten", "23.12.2013", "31.12.2013"),
    
    # 2014
    ("Weihnachten", "01.01.2014", "06.01.2014"),
    ("Ostern", "16.04.2014", "02.05.2014"),
    ("Pfingsten", "30.05.2014", "30.05.2014"),
    ("Sommer", "14.07.2014", "23.08.2014"),
    ("Herbst", "13.10.2014", "25.10.2014"),
    ("Weihnachten", "22.12.2014", "31.12.2014"),
    
    # 2015
    ("Weihnachten", "01.01.2015", "06.01.2015"),
    ("Ostern", "01.04.2015", "17.04.2015"),
    ("Pfingsten", "15.05.2015", "15.05.2015"),
    ("Sommer", "20.07.2015", "29.08.2015"),
    ("Herbst", "19.10.2015", "31.10.2015"),
    ("Weihnachten", "21.12.2015", "31.12.2015"),
    
    # 2016
    ("Weihnachten", "01.01.2016", "06.01.2016"), 
    ("Ostern", "24.03.2016", "09.04.2016"),
    ("Pfingsten", "06.05.2016", "06.05.2016"),
    ("Sommer", "25.07.2016", "03.09.2016"),
    ("Herbst", "17.10.2016", "29.10.2016"),
    ("Weihnachten", "23.12.2016", "31.12.2016"),
    
    # 2017
    ("Weihnachten", "01.01.2017", "06.01.2017"),
    ("Ostern", "07.04.2017", "21.04.2017"),
    ("Pfingsten", "26.05.2017", "26.05.2017"),
    ("Sommer", "24.07.2017", "02.09.2017"),
    ("Herbst", "16.10.2017", "27.10.2017"),
    ("Weihnachten", "21.12.2017", "31.12.2017"),
    
    # 2018
    ("Weihnachten", "01.01.2018", "06.01.2018"),
    ("Ostern", "29.03.2018", "13.04.2018"),
    ("Pfingsten", "11.05.2018", "11.05.2018"),
    ("Sommer", "09.07.2018", "18.08.2018"),
    ("Herbst", "01.10.2018", "19.10.2018"),
    ("Weihnachten", "21.12.2018", "31.12.2018")
]

# Create lists to store expanded data
dates = []
holiday_names = []

# Process each holiday period
for holiday_name, start_date_str, end_date_str in holiday_periods:
    # Convert string dates to datetime objects
    start_date = datetime.strptime(start_date_str, "%d.%m.%Y")
    end_date = datetime.strptime(end_date_str, "%d.%m.%Y")
    
    # Get all dates in the range
    date_range = expand_date_range(start_date, end_date)
    
    # Add to lists
    dates.extend(date_range)
    holiday_names.extend([holiday_name] * len(date_range))

# Create DataFrame
df = pd.DataFrame({
    'Datum': dates,
    'school_holiday_name': holiday_names,
    'is_school_holiday': 1
})

# Sort by date
df = df.sort_values('Datum')

# Save to CSV
df.to_csv('data/school_holidays.csv', index=False)

# Print summary statistics
print("\nSchool Holidays Summary:")
print(f"Total number of holiday days: {len(df)}")
print(f"Date range: from {df['Datum'].min().date()} to {df['Datum'].max().date()}")
print("\nHoliday types:")
for holiday in sorted(df['school_holiday_name'].unique()):
    days = len(df[df['school_holiday_name'] == holiday])
    print(f"- {holiday}: {days} days")

print("\nCSV file 'german_school_holidays.csv' has been created with columns:")
print("- date")
print("- school_holiday_name")
print("- is_school_holiday")