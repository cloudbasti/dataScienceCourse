import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.feature_prep import prepare_features
from data.train_split import split_train_validation



def prepare_and_predict_umsatz(df):
    """
    Main function that uses the preparation functions and fits the linear regression model.
    """
    # Define features
    feature_columns = ['Temperatur', 'Bewoelkung', 'Wochentag_encoded']
    
    # Prepare features
    df_cleaned, label_encoder = prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_train_validation(df_cleaned, feature_columns)
    
    # Create and fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Create a dictionary of feature coefficients
    feature_coefficients = dict(zip(feature_columns, model.coef_))
    
    return model, feature_coefficients, r2, rmse, label_encoder, y_test, y_pred


# Load and merge data
weather = pd.read_csv("data/wetter.csv")
kiwo = pd.read_csv("data/kiwo.csv")
turnover = pd.read_csv("data/umsatzdaten_gekuerzt.csv")
df = pd.merge(weather, turnover, on='Datum', how='outer')

# Train the model
model, coefficients, r2, rmse, label_encoder, y_test, y_pred = prepare_and_predict_umsatz(df)

# Print results
print("\nModel Performance:")
print(f"R-squared score: {r2:.3f}")
print(f"Root Mean Squared Error: {rmse:.2f}")

print("\nFeature Coefficients:")
for feature, coef in coefficients.items():
    print(f"{feature}: {coef:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# Example of making a new prediction for a specific day
new_data = pd.DataFrame({
    'Temperatur': [24],
    'Bewoelkung': [3],
    'Wochentag_encoded': [label_encoder.transform(['Wednesday'])]
})

predicted_umsatz = model.predict(new_data)
print(f"\nPredicted turnover for Wednesday with Temp=24 and Bewoelkung=3: {predicted_umsatz[0]:.2f}")