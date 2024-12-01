import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

weather = pd.read_csv("wetter.csv")
kiwo = pd.read_csv("kiwo.csv")
turnover = pd.read_csv("umsatzdaten_gekuerzt.csv")

df = pd.merge(weather, turnover, on='Datum', how='outer')

df.head()

# check for missing values
df.isnull().sum()

# (for this model to work we need to remove NaN values)
# Create a copy to avoid modifying the original DataFrame
df_prepared = df.copy()
    
# Drop rows where either temperature or turnover is missing
df_prepared = df_prepared.dropna(subset=['Temperatur', 'Umsatz'])

def predict_turnover(df_prepared, weather_feature='Temperatur'):
    
    # Prepare the data
    X = df_prepared[[weather_feature]]
    y = df_prepared['Umsatz']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return model, r2, rmse


model, r2, rmse = predict_turnover(df_prepared, weather_feature='Temperatur')

# Print results
print(f"R-squared score: {r2:.3f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Coefficient: {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# Make a new prediction
new_weather = pd.DataFrame({'Temperatur': [24]})
predicted_turnover = model.predict(new_weather)
print(f"Predicted turnover for temperature 24°C: {predicted_turnover[0]:.2f}")