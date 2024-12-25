import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.train_split import split_train_validation
from data.data_prep import prepare_features
from data.data_prep import merge_datasets
from data.data_prep import handle_missing_values

def create_product_features(df):
    """
    Create product-specific features, indicators, and interactions.
    """
    df_with_features = df.copy()
    base_features = ['Temperatur', 'Bewoelkung', 'Wochentag_encoded', 'is_holiday', 
                    'is_school_holiday', 'KielerWoche', 'is_weekend']
    all_features = []
    
    # Create product indicators and interactions
    for product_id in range(1, 7):
        # Create product indicator (1 if this product, 0 otherwise)
        product_col = f'is_product_{product_id}'
        df_with_features[product_col] = (df_with_features['Warengruppe'] == product_id).astype(int)
        all_features.append(product_col)
        
        # Create interaction terms
        for feature in base_features:
            interaction_col = f'{feature}_product_{product_id}'
            df_with_features[interaction_col] = df_with_features[product_col] * df_with_features[feature]
            all_features.append(interaction_col)
    
    return df_with_features, all_features

def prepare_and_predict_umsatz_nn(df):
    """
    Neural network model for predicting turnover with product interactions
    """
    # Create product features and interactions
    df_with_features, feature_columns = create_product_features(df)
    
    # Split data using our existing function
    X_train, X_test, y_train, y_test = split_train_validation(df_with_features, feature_columns)
    
    # Scale the features and target
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()
    
    # Create the neural network
    model = Sequential([
        Dense(128, activation='relu', input_shape=(len(feature_columns),)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(), 
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='mse',
                 metrics=['mae'])
    
    early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=30,
    restore_best_weights=True
)
    
    # Train the model
    history = model.fit(X_train_scaled, y_train_scaled,
                       epochs=200,
                       batch_size=32,
                       validation_split=0.2,
                       callbacks=[early_stopping],
                       verbose=1)
    
    # Make predictions
    y_pred_scaled = model.predict(X_test_scaled)
    
    # Inverse transform predictions and actual values
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_unscaled = y_test.values.reshape(-1, 1)
    
    # Calculate overall metrics
    r2 = r2_score(y_test_unscaled, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred))
    
    # Calculate metrics per product category
    product_metrics = {}
    for product_id in range(1, 7):
        mask = X_test[f'is_product_{product_id}'] == 1
        if mask.any():
            product_r2 = r2_score(y_test_unscaled[mask], y_pred[mask])
            product_rmse = np.sqrt(mean_squared_error(y_test_unscaled[mask], y_pred[mask]))
            product_metrics[f"Product {product_id}"] = {'R2': product_r2, 'RMSE': product_rmse}
    
    return model, r2, rmse, product_metrics, scaler_X, scaler_y, history, feature_columns

def plot_training_history(history):
    """Plot training and validation loss"""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def main():
    # Load and prepare data
    df_merged = merge_datasets()
    df_featured, weekday_encoder = prepare_features(df_merged)
    df_cleaned = handle_missing_values(df_featured)
    
    # Train neural network
    model, r2, rmse, product_metrics, scaler_X, scaler_y, history, feature_columns = prepare_and_predict_umsatz_nn(df_cleaned)
    
    # Print overall results
    print("\nNeural Network Overall Performance:")
    print(f"R-squared score: {r2:.3f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    
    # Print performance by product category
    print("\nNeural Network Performance by Product Category:")
    for category, metrics in product_metrics.items():
        print(f"\n{category}:")
        print(f"R-squared: {metrics['R2']:.3f}")
        print(f"RMSE: {metrics['RMSE']:.2f}")
    
    # Example predictions
    print("\nExample Predictions:")
    for product_id in range(1, 7):
        # Create new data frame with product indicators
        new_data = pd.DataFrame({f'is_product_{i}': [1 if i == product_id else 0] for i in range(1, 7)})
        
        # Add interaction terms
        base_values = {
            'Temperatur': 24,
            'Bewoelkung': 3,
            'Wochentag_encoded': weekday_encoder.transform(['Wednesday'])[0],
            'is_holiday': 0,
            'is_school_holiday': 1,
            'KielerWoche': 1,
            'is_weekend': 0
        }
        
        # Create interaction terms
        for i in range(1, 7):
            for feature, value in base_values.items():
                new_data[f'{feature}_product_{i}'] = new_data[f'is_product_{i}'] * value
        
        # Ensure columns are in the right order
        example_data = new_data[feature_columns]
        
        # Scale and predict
        example_scaled = scaler_X.transform(example_data)
        pred_scaled = model.predict(example_scaled)
        pred = scaler_y.inverse_transform(pred_scaled)
        
        print(f"\nPredicted turnover for Product {product_id}: {pred[0][0]:.2f}")
    
    # Plot training history
    plot_training_history(history)

if __name__ == "__main__":
    main()