import pandas as pd

df = pd.read_csv("wetter.csv")

print(df.head())

# extract temperature column and find average temperature

temp = df["Temperatur"]

print(temp.mean())

# average temperature of July and May
df['Datum'] = pd.to_datetime(df['Datum'])

july = df[ (df['Datum'].dt.month == 7)]

may = df[ (df['Datum'].dt.month == 5)]

julyTemp = july["Temperatur"]
mayTemp = may["Temperatur"]

julyAverage = julyTemp.mean()
mayAverage = mayTemp.mean()

print(julyAverage)
print(mayAverage)





