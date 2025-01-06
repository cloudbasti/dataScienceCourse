# dataScienceCourse

# to do: 
MOST IMPORTANT: 
1. Time Series Prediction als Features hinzufügen
2. Learning Rate Optimization
3. Dropout Layer Optimization
4. nochmal über die Interaction Features nachdenken und eventuell umsortieren


Statistiken und Analyse: 
1. WetterCodes mit 0 machen eventuell keinen Sinn

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

