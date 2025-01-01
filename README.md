# dataScienceCourse

# to do: 

DATENERGÄNZUNG
1. add Brückentage before and after holidays

MEHR FEATURES, basierend auch auf neuen Daten

1. Wettercode (hier nochmal die Hauptkategorien besser beschreiben)
2. Windgeschwindigkeit (hier ebenfalls)
3. Feiertage (Schulferien und sowieso Feiertag?)

Read the Task Descriptions of the other weeks and do all tasks. 

4. See if you can upload Data on Kaggle


VERBESSERUNG NEURONALES NETZ (ein erstes Neuronales Netz optimieren)
1. Learning Rate optimization
2. Prevent Overfitting
3. Batch Optimization
4. Anzahl Neurons und Dropout Layer

First try with a shallow model with few neurons. Then gradually increase the number of neurons in order to overfit the dataset. You should be able to reach perfect score on the train set. At that point you can start decrease the number of neurons and add layers with dropout to improve generalisation.

questions for AI: 
1. What are good values for MSE and MAE? Also print these!
2. what are the steps to train my network? 
3. Should you first increase the learning rate gradually or change the neurons and see when overfitting occurs? 
4. what are useful diagrams for model performance and convergence? I need plots of these? 

SOLUTIONS: 

for the diagrams you need to make diagrams for the training and validation set based on the trained model. Also implement the MAPE, as in the notebooks. 


DATA ANALYSIS AND DATA VALUES: 

Outliers:
1. analyze the data for outliers
2. what are weird values? 
3. if a weather code is set to zero if its not existing, does this really make sense?

Missing Values
1. Understand the data better with respect to missing values
2. How are missing values handled for the different features? 
-- replace the values with new values --
3. Imputation of values (just implement one strategy)


