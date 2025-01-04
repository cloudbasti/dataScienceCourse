# dataScienceCourse

# to do: 
MOST IMPORTANT: 
1. Add Bank holidays and holidays to create_bank/school_holidays 
2. Data Imputation / vielleicht unterschiedlich für Training Set und Final Set
3. Time Series Prediction als Features hinzufügen

Optimierung: 
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


