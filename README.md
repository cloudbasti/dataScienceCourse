# dataScienceCourse

# to do: 
MOST IMPORTANT: 
3. Time Series Prediction als Features hinzufügen

Optimierung: 
3. nochmal über die Interaction Features nachdenken und eventuell umsortieren
4. Cyclical month versuchen
4. Optimieren des neuronalen Netzes
5. Training Loss / Validation Loss / MAPE

Statistiken und Analyse: 
1. Deskriptive Statistiken für zwei Variablen
2. what are weird values? 
3. if a weather code is set to zero if its not existing, does this really make sense?

Optional: 
4. weitere Features hinzufügen
5. add Brückentage before and after holidays

weitere Optional
1. Wettercode (hier nochmal die Hauptkategorien besser beschreiben)
2. Windgeschwindigkeit (hier ebenfalls)
3. Feiertage (Schulferien und sowieso Feiertag?)


First try with a shallow model with few neurons. Then gradually increase the number of neurons in order to overfit the dataset. You should be able to reach perfect score on the train set. At that point you can start decrease the number of neurons and add layers with dropout to improve generalisation.

questions for AI: 
1. What are good values for MSE and MAE? Also print these!
2. what are the steps to train my network? 
3. Should you first increase the learning rate gradually or change the neurons and see when overfitting occurs? 
4. what are useful diagrams for model performance and convergence? I need plots of these? 

SOLUTIONS: 

for the diagrams you need to make diagrams for the training and validation set based on the trained model. Also implement the MAPE, as in the notebooks. 



Ohne Imputation:
Neural Network Overall Performance:
R-squared score: 0.850
Root Mean Squared Error: 50.60

Neural Network Performance by Product Category:

Product 1:
R-squared: 0.312
RMSE: 35.31

Product 2:
R-squared: 0.715
RMSE: 68.26

Product 3:
R-squared: 0.692
RMSE: 42.42

Product 4:
R-squared: 0.197
RMSE: 23.80

Product 5:
R-squared: 0.397
RMSE: 68.93

Product 6:
R-squared: 0.023
RMSE: 30.57


