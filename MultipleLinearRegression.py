import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import numpy as np

weather = pd.read_csv("wetter.csv")
kiwo = pd.read_csv("kiwo.csv")
turnover = pd.read_csv("umsatzdaten_gekuerzt.csv")

df = pd.merge(weather, turnover, on='Datum', how='outer')

def prepare_and_predict_umsatz(df):
    # Create a copy to avoid modifying the original
    df_prepared = df.copy()
    
    # Convert date and add weekday
    df_prepared['Datum'] = pd.to_datetime(df_prepared['Datum'])
    df_prepared['Wochentag'] = df_prepared['Datum'].dt.day_name()
    
    # Convert weekdays to numerical values
    label_encoder = LabelEncoder()
    df_prepared['Wochentag_encoded'] = label_encoder.fit_transform(df_prepared['Wochentag'])
    
    # Define features
    feature_columns = ['Temperatur', 'Bewoelkung', 'Wochentag_encoded']
    
    # Drop rows with missing values
    columns_to_check = ['Temperatur', 'Bewoelkung', 'Umsatz']
    df_cleaned = df_prepared.dropna(subset=columns_to_check)
    
    # Print the number of rows removed
    rows_removed = len(df_prepared) - len(df_cleaned)
    print(f"Removed {rows_removed} rows with missing values")
    
    # Prepare the data
    X = df_cleaned[feature_columns]
    y = df_cleaned['Umsatz']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
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
    
    return model, feature_coefficients, r2, rmse, label_encoder


# Train the model
model, coefficients, r2, rmse, label_encoder = prepare_and_predict_umsatz(df)

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