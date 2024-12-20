import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

weather = pd.read_csv("wetter.csv")
kiwo = pd.read_csv("kiwo.csv")
turnover = pd.read_csv("umsatzdaten_gekuerzt.csv")

df = pd.merge(weather, turnover, on='Datum', how='outer')

def prepare_and_predict_umsatz_nn(df):
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
    
    # Scale the features and target
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    
    # Create the neural network
    model = Sequential([
        Dense(64, activation='relu', input_shape=(len(feature_columns),)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='mse',
                 metrics=['mae'])
    
    # Train the model
    history = model.fit(X_train, y_train,
                       epochs=100,
                       batch_size=32,
                       validation_split=0.2,
                       verbose=1)
    
    # Make predictions
    y_pred_scaled = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_unscaled = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate metrics
    r2 = r2_score(y_test_unscaled, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred))
    
    return model, r2, rmse, scaler_X, scaler_y, label_encoder, history

# Train the model
model, r2, rmse, scaler_X, scaler_y, label_encoder, history = prepare_and_predict_umsatz_nn(df)

# Print results
print("\nNeural Network Performance:")
print(f"R-squared score: {r2:.3f}")
print(f"Root Mean Squared Error: {rmse:.2f}")

# Example of making a new prediction for a specific day
new_data = pd.DataFrame({
    'Temperatur': [24],
    'Bewoelkung': [3],
    'Wochentag_encoded': [label_encoder.transform(['Wednesday'])]
})

# Scale the new data
new_data_scaled = scaler_X.transform(new_data)

# Make prediction and inverse transform
predicted_umsatz_scaled = model.predict(new_data_scaled)
predicted_umsatz = scaler_y.inverse_transform(predicted_umsatz_scaled)

print(f"\nPredicted turnover for Wednesday with Temp=24 and Bewoelkung=3: {predicted_umsatz[0][0]:.2f}")

# Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()