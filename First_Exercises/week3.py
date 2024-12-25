import pandas as pd

weather = pd.read_csv("wetter.csv")
kiwo = pd.read_csv("kiwo.csv")
turnover = pd.read_csv("umsatzdaten_gekuerzt.csv")

df = pd.merge(weather, turnover, on='Datum', how='outer')

kiwo['Datum'] = pd.to_datetime(kiwo['Datum'])
df['Datum'] = pd.to_datetime(df['Datum'])

filter_date = '2013-07-01'

filtered_df = df[df['Datum'] >= filter_date]


# print object types to see if they are the same
#print(df.dtypes)
#print(kiwo.dtypes)

final_merge = pd.merge(df, kiwo, on='Datum', how='outer')

filter_date = '2019-06-26'

filtered_df = final_merge[final_merge['Datum'] >= filter_date]

print(filtered_df.head(10))
#print(weather)
print(kiwo)
#print(turnover)
