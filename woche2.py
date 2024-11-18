import pandas as pd

df = pd.read_csv("umsatzdaten_gekuerzt.csv")

# filter out by weekdays

df['Datum'] = pd.to_datetime(df['Datum'])

df['Tag'] = df['Datum'].dt.day_name()

print(df)