import pandas as pd
import numpy as np

import scipy.stats as stats

df = pd.read_csv("wetter.csv")

# extract temperature column and find average temperature

temp = df["Temperatur"]

print("mean temperature:", temp.mean())

# average temperature of July and May
df['Datum'] = pd.to_datetime(df['Datum'])

# filter out july and may from the dataset
july = df[ (df['Datum'].dt.month == 7)]
may = df[ (df['Datum'].dt.month == 5)]

julyAverage = july["Temperatur"].mean()
mayAverage = may["Temperatur"].mean()

print("july average:", julyAverage)
print("may average:", mayAverage)

# standard deviation, confidence intervals


# standard deviation and sample size
std_dev_july = july["Temperatur"].std()
n_july = july["Temperatur"].count()

std_dev_may = may["Temperatur"].std()
n_may = may["Temperatur"].count()

# standard error of the mean

sem_may = std_dev_may / np.sqrt(n_may)
sem_july = std_dev_july / np.sqrt(n_july)


alpha = 0.05
critical_value = stats.norm.ppf(1 - alpha/2)

# Calculate the confidence interval
margin_of_error_may = critical_value * sem_may
margin_of_error_july = critical_value * sem_july
confidence_interval_may = (mayAverage - margin_of_error_may, mayAverage + margin_of_error_may)
confidence_interval_july = (julyAverage - margin_of_error_july, julyAverage + margin_of_error_july)






