import pandas as pd

# Create lists of holidays and dates
holidays = [
    # 2013
    ('Neujahr', '2013-01-01'),
    ('Karfreitag', '2013-03-29'),
    ('Ostermontag', '2013-04-01'),
    ('Tag der Arbeit', '2013-05-01'),
    ('Christi Himmelfahrt', '2013-05-09'),
    ('Pfingstmontag', '2013-05-20'),
    ('Tag der Deutschen Einheit', '2013-10-03'),
    ('Reformationstag', '2013-10-31'),
    ('Weihnachten', '2013-12-25'),
    ('2. Weihnachtstag', '2013-12-26'),

    # 2014
    ('Neujahr', '2014-01-01'),
    ('Karfreitag', '2014-04-18'),
    ('Ostermontag', '2014-04-21'),
    ('Tag der Arbeit', '2014-05-01'),
    ('Christi Himmelfahrt', '2014-05-29'),
    ('Pfingstmontag', '2014-06-09'),
    ('Tag der Deutschen Einheit', '2014-10-03'),
    ('Reformationstag', '2014-10-31'),
    ('Weihnachten', '2014-12-25'),
    ('2. Weihnachtstag', '2014-12-26'),

    # 2015
    ('Neujahr', '2015-01-01'),
    ('Karfreitag', '2015-04-03'),
    ('Ostermontag', '2015-04-06'),
    ('Tag der Arbeit', '2015-05-01'),
    ('Christi Himmelfahrt', '2015-05-14'),
    ('Pfingstmontag', '2015-05-25'),
    ('Tag der Deutschen Einheit', '2015-10-03'),
    ('Reformationstag', '2015-10-31'),
    ('Weihnachten', '2015-12-25'),
    ('2. Weihnachtstag', '2015-12-26'),

    # 2016
    ('Neujahr', '2016-01-01'),
    ('Karfreitag', '2016-03-25'),
    ('Ostermontag', '2016-03-28'),
    ('Tag der Arbeit', '2016-05-01'),
    ('Christi Himmelfahrt', '2016-05-05'),
    ('Pfingstmontag', '2016-05-16'),
    ('Tag der Deutschen Einheit', '2016-10-03'),
    ('Reformationstag', '2016-10-31'),
    ('Weihnachten', '2016-12-25'),
    ('2. Weihnachtstag', '2016-12-26'),

    # 2017
    ('Neujahr', '2017-01-01'),
    ('Karfreitag', '2017-04-14'),
    ('Ostermontag', '2017-04-17'),
    ('Tag der Arbeit', '2017-05-01'),
    ('Christi Himmelfahrt', '2017-05-25'),
    ('Pfingstmontag', '2017-06-05'),
    ('Tag der Deutschen Einheit', '2017-10-03'),
    ('Reformationstag', '2017-10-31'),
    ('Weihnachten', '2017-12-25'),
    ('2. Weihnachtstag', '2017-12-26'),

    # 2018
    ('Neujahr', '2018-01-01'),
    ('Karfreitag', '2018-03-30'),
    ('Ostermontag', '2018-04-02'),
    ('Tag der Arbeit', '2018-05-01'),
    ('Christi Himmelfahrt', '2018-05-10'),
    ('Pfingstmontag', '2018-05-21'),
    ('Tag der Deutschen Einheit', '2018-10-03'),
    ('Reformationstag', '2018-10-31'),
    ('Weihnachten', '2018-12-25'),
    ('2. Weihnachtstag', '2018-12-26'),

    # 2019
    ('Neujahr', '2019-01-01'),
    ('Karfreitag', '2019-04-19'),
    ('Ostermontag', '2019-04-22'),
    ('Tag der Arbeit', '2019-05-01'),
    ('Christi Himmelfahrt', '2019-05-30'),
    ('Pfingstmontag', '2019-06-10'),
    ('Tag der Deutschen Einheit', '2019-10-03'),
    ('Reformationstag', '2019-10-31'),
    ('Weihnachten', '2019-12-25'),
    ('2. Weihnachtstag', '2019-12-26')
]

# Create DataFrame
df = pd.DataFrame(holidays, columns=['holiday_name', 'Datum'])

# Convert date to datetime
df['Datum'] = pd.to_datetime(df['Datum'])

# Sort by date
df = df.sort_values('Datum')


# Add is_holiday column (all 1s since these are all holidays)
df['is_holiday'] = 1

# Display the first few rows to verify
print("\nFirst few holidays:")
print(df.head())

# Save to CSV
df.to_csv('data/HolidayData/bank_holidays.csv', index=False)

# Display some basic statistics
print("\nSummary:")
print(f"Total number of holidays: {len(df)}")
print(f"Date range: from {df['Datum'].min().date()} to {
      df['Datum'].max().date()}")
print(f"\nUnique holidays ({len(df['holiday_name'].unique())} total):")
for holiday in sorted(df['holiday_name'].unique()):
    print(f"- {holiday}")

print("\nCSV file 'german_bank_holidays.csv' has been created with columns:")
print("- holiday_name")
print("- date")
print("- is_holiday")
