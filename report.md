Fehlende Werte bei den normalen WetterDaten: 

Missing Values Summary:
                          Data Type  Missing Values  Percentage Missing
Datum                datetime64[ns]               0                0.00
Bewoelkung                  float64              10                0.38
Temperatur                  float64               0                0.00
Windgeschwindigkeit           int64               0                0.00
Wettercode                  float64             669               25.72


Missing values by year:
      Datum  Bewoelkung  Temperatur  Windgeschwindigkeit  Wettercode  year  month  day_of_week
year                                                                                          
2012      0           0           0                    0         129     0      0            0
2013      0           0           0                    0          89     0      0            0
2014      0           0           0                    0         129     0      0            0
2015      0           0           0                    0          97     0      0            0
2016      0           0           0                    0          88     0      0            0
2017      0          10           0                    0          64     0      0            0
2018      0           0           0                    0          39     0      0            0
2019      0           0           0                    0          34     0      0            0

Missing values by month:
       Datum  Bewoelkung  Temperatur  Windgeschwindigkeit  Wettercode  year  month  day_of_week
month                                                                                          
1          0           0           0                    0          26     0      0            0
2          0           0           0                    0          28     0      0            0
3          0           0           0                    0          46     0      0            0
4          0           0           0                    0          53     0      0            0
5          0           0           0                    0          90     0      0            0
6          0           0           0                    0          91     0      0            0
7          0           0           0                    0          91     0      0            0
8          0           0           0                    0          79     0      0            0
9          0           0           0                    0          68     0      0            0
10         0           2           0                    0          41     0      0            0
11         0           8           0                    0          34     0      0            0
12         0           0           0                    0          22     0      0            0

Missing values by day of week:
             Datum  Bewoelkung  Temperatur  Windgeschwindigkeit  Wettercode  year  month  day_of_week
day_of_week                                                                                          
0                0           2           0                    0         103     0      0            0
1                0           2           0                    0          97     0      0            0
2                0           2           0                    0          99     0      0            0
3                0           1           0                    0          96     0      0            0
4                0           1           0                    0          87     0      0            0
5                0           1           0                    0          88     0      0            0
6                0           1           0                    0          99     0      0            0

Most common weather codes:
Wettercode
61.0    594
21.0    260
0.0     212
10.0    189
5.0     163
63.0    131
20.0     68
95.0     41
80.0     33
71.0     26


WETTERCODE Imputation
------------------------------------------------------------
            Before  After  Difference
Wettercode                           
61.0            42     68          26
21.0            29     46          17
0.0             25     41          16
5.0             26     41          15
63.0            11     17           6

Month 6:
------------------------------------------------------------
            Before  After  Difference
Wettercode                           
61.0            44     73          29
21.0            27     43          16
0.0             20     32          12
95.0            15     24           9
5.0             12     20           8

Month 7:
------------------------------------------------------------
            Before  After  Difference
Wettercode                           
61.0            41     67          26
5.0             25     40          15
21.0            25     39          14
0.0             16     25           9
63.0            12     20           8

Before and After Imputation Neural Network
Neural Network Overall Performance:
R-squared score: 0.864
Root Mean Squared Error: 48.15

Neural Network Performance by Product Category:

Product 1:
R-squared: 0.304
RMSE: 35.49

Product 2:
R-squared: 0.790
RMSE: 58.60

Product 3:
R-squared: 0.694
RMSE: 42.28

Product 4:
R-squared: 0.034
RMSE: 26.11

Product 5:
R-squared: 0.409
RMSE: 68.22

Product 6:
R-squared: 0.297
RMSE: 25.93

AFTER: 
Neural Network Overall Performance:
R-squared score: 0.870
Root Mean Squared Error: 47.18

Neural Network Performance by Product Category:

Product 1:
R-squared: 0.295
RMSE: 35.72

Product 2:
R-squared: 0.769
RMSE: 61.48

Product 3:
R-squared: 0.702
RMSE: 41.78

Product 4:
R-squared: 0.124
RMSE: 24.86

Product 5:
R-squared: 0.504
RMSE: 62.50

Product 6:
R-squared: 0.185
RMSE: 27.92

Neural Network Overall Performance:
R-squared score: 0.865
Root Mean Squared Error: 48.03

Neural Network Performance by Product Category:

Product 1:
R-squared: 0.324
RMSE: 34.99

Product 2:
R-squared: 0.769
RMSE: 61.40

Product 3:
R-squared: 0.698
RMSE: 42.04

Product 4:
R-squared: -0.113
RMSE: 28.02

Product 5:
R-squared: 0.467
RMSE: 64.81

Product 6:
R-squared: 0.206
RMSE: 27.55

Effekt so direkt erstmal nicht sichtbar.

Präsentation (Powerpoint, Keybote oder ähnliches)
Jedes Team hält eine 8 oder 10-minütige Abschlusspräsentation (genaue Info erfolgt in der Vorwoche - bitte darauf achten, dass Ihr die Länge einhaltet!) mit den folgenden Inhalten:

# Euren Namen auf der Titelseite DONE

# Auflistung und kurze Beschreibung der selbst erstellten Variablen DONE

Balkendiagramme mit Konfidenzintervallen für zwei selbst erstellte Variablen

Optimierung des linearen Modells: Modellgleichung und adjusted r²

Art der Missing Value Imputation

Optimierung des neuronalen Netzes:

Source Code zur Definition des neuronalen Netzes

Darstellung der Loss-Funktionen für Trainings- und Validierungsdatensatz

MAPEs für den Validierungsdatensatz insgesamt und für jede Warengruppe einzeln

„Worst Fail“ / „Best Improvement“