# Dataset Analysis

This document provides a comprehensive analysis of our dataset collection, including information about data distributions, missing values, and potential biases.

## 1. Dataset Overview

Our dataset consists of three main CSV files containing different types of data.

### Sample Counts
```
Dataset Overview - Sample Counts:
---------------------------------
data/OriginalData/umsatzdaten_gekuerzt.csv: 9334 samples (Date Range: 2013-07-01 to 2018-07-31)
data/OriginalData/wetter.csv: 2601 samples (Date Range: 2012-01-01 to 2019-08-01)
data/OriginalData/kiwo.csv: 72 samples (Date Range: 2012-06-16 to 2019-06-30)
---------------------------------
```

### Product Category Distribution
```
---------------------------------
Warengruppe 1: 1819 samples (Date Range: 2013-07-01 to 2018-07-31)
Warengruppe 2: 1819 samples (Date Range: 2013-07-01 to 2018-07-31)
Warengruppe 3: 1819 samples (Date Range: 2013-07-01 to 2018-07-31)
Warengruppe 4: 1766 samples (Date Range: 2013-07-01 to 2018-07-31)
Warengruppe 5: 1819 samples (Date Range: 2013-07-01 to 2018-07-31)
Warengruppe 6: 292 samples (Date Range: 2013-10-24 to 2017-12-27)
Total: 9334 samples
---------------------------------
```

## 2. Handling Missing Values

```
Missing Values Analysis:
---------------------------------
umsatzdaten_gekuerzt.csv: 0 missing values out of 28002 cells (0.00%) - Total rows: 9334
wetter.csv: 679 missing values out of 13005 cells (5.22%) - Total rows: 2601 (Bewoelkung: 10, Wettercode: 669)
kiwo.csv: 0 missing values out of 144 cells (0.00%) - Total rows: 72
---------------------------------
```

Our approach to handling missing values:

1. **Temperature**: Missing values are imputed using weekly averages.
2. **Wind speed**: Missing values are filled with the median wind speed.
3. **Cloud cover (Bewoelkung)**: Missing values are filled with the rounded median.
4. **Weather code (Wettercode)**: Missing values are filled with randomly sampled values based on the monthly distribution.
5. **KielerWoche**: Missing values are filled with 0 (indicating no Kieler Woche event).
6. **Holiday indicators**: Missing values in school and public holiday columns are filled with 0.
7. **Time-series lags**: Missing values in lag features are filled with the median turnover for the respective product group.
8. **Feature engineering**: Missing values in derived features are filled with appropriate defaults (typically 0).


## 3. Feature Distributions

Here are just some distributions to indicate the effect of some features. 
### Turnover Distribution Analysis
```
Turnover Distribution Analysis
==============================
                     Comparison           Group 1 Turnover 1      Group 2 Turnover 2 Difference Ratio
             Weekday vs Weekend           Weekday     191.80      Weekend     243.91      52.12  1.27
 New Year's Eve vs Regular Days    New Year's Eve     596.50 Regular Days     205.79     390.71  2.90
Last Day of Month vs Other Days Last Day of Month     241.71   Other Days     205.58      36.13  1.18
    July/August vs Other Months       July/August     279.43 Other Months     191.03      88.41  1.46
```

### Key Features Used in Analysis (excluding lag features)

1. **Umsatz**: Daily turnover (target variable)
2.  **weather categories**: Various weather condition indicators (weather_clear, weather_rain, etc.)
3. **Warengruppe**: Product group/category
4. **Temperatur**: Temperature in Celsius
5. **Windgeschwindigkeit**: Wind speed
6. **Wettercode**: Weather code
7. **Bewoelkung**: Cloud cover
8. **KielerWoche**: Binary indicator for Kieler Woche festival
9. **is_holiday**: Binary indicator for public holidays
10. **is_school_holiday**: Binary indicator for school holidays
11. **is_nye**: Binary indicator for New Year's Eve
12. **is_christmas_eve**: Binary indicator for Christmas Eve
13. **is_last_day_of_month**: Binary indicator for last day of month
14. **Wochentag**: Day of the week
15. **is_weekend**: Binary indicator for weekend days
16. **month**: Month of the year
17. **season**: Season (Winter, Spring, Summer, Autumn)
18. **is_pre_holiday**: Binary indicator for day before a holiday
19. **is_peak_summer**: Binary indicator for peak summer months (July, August)
