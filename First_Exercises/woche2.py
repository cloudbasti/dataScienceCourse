import pandas as pd

df = pd.read_csv("umsatzdaten_gekuerzt.csv")

# add weekdays to dataset (not needed in fact but its needed for the 
# other solution with the for loop)

df['Datum'] = pd.to_datetime(df['Datum'])

df['Tag'] = df['Datum'].dt.day_name()

# calculate average turnover per weekday

monday = df[ (df['Datum'].dt.weekday == 0)]
mondayAverage = monday["Umsatz"].mean()

tuesday = df[ (df['Datum'].dt.weekday == 1)]
tuesdayAverage = tuesday["Umsatz"].mean()

wednesday = df[ (df['Datum'].dt.weekday == 2)]
wednesdayAverage = wednesday["Umsatz"].mean()

thursday = df[ (df['Datum'].dt.weekday == 3)]
thursdayAverage = thursday["Umsatz"].mean()

friday = df[ (df['Datum'].dt.weekday == 4)]
fridayAverage = friday["Umsatz"].mean()

saturday = df[ (df['Datum'].dt.weekday == 5)]
saturdayAverage = saturday["Umsatz"].mean()

sunday = df[ (df['Datum'].dt.weekday == 6)]
sundayAverage = sunday["Umsatz"].mean()

# store results in array to pass to matplotlib



# graphical visualisation
import matplotlib.pyplot as plt

Wochentag = ('Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag', 'Sonntag')
Umsatz = [mondayAverage, tuesdayAverage, wednesdayAverage, thursdayAverage, fridayAverage, saturdayAverage, sundayAverage]
 
plt.bar(Wochentag, Umsatz, align='center')

plt.show()