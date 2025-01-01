import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.train_split import split_train_validation
from data.data_prep import prepare_features
from data.data_prep import merge_datasets
from data.data_prep import handle_missing_values

def create_callbacks():
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=50,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=50,
        min_lr=0.00001,
        verbose=1
    )
    
    return [early_stopping, reduce_lr]

def create_product_features(df):
    df_with_features = df.copy()
    
    weather_features = ['Temperatur', 'Bewoelkung']
    weather_codes = [col for col in df.columns if col.startswith('weather_')]
    wind_features = [col for col in df.columns if col.startswith('wind_')]
    
    weekday_dummies = [col for col in df.columns if col.startswith('weekday_')]
    month_dummies = [col for col in df.columns if col.startswith('month_')]
    season_dummies = [col for col in df.columns if col.startswith('season_')]
   
    time_features = weekday_dummies + month_dummies + season_dummies
    event_features = ['is_holiday', 'is_school_holiday', 'KielerWoche', 'is_nye']
    
    all_features = []
    
    # Temperature polynomials
    temp_poly = PolynomialFeatures(degree=3, include_bias=False)
    temp_features = temp_poly.fit_transform(df_with_features[['Temperatur']])
    feature_names = ['Temp', 'Temp2', 'Temp3']
    
    for i, name in enumerate(feature_names):
        df_with_features[name] = temp_features[:, i]
        all_features.append(name)
    
    # Weather interactions
    df_with_features['Temp_Cloud'] = df_with_features['Temperatur'] * df_with_features['Bewoelkung']
    all_features.append('Temp_Cloud')
    
    all_features.extend(weekday_dummies)
    all_features.extend(month_dummies)
    all_features.extend(season_dummies)
    all_features.extend(weather_codes)
    all_features.extend(wind_features)
    
    # Product-specific features
    for product_id in range(1, 7):
        product_col = f'is_product_{product_id}'
        df_with_features[product_col] = (df_with_features['Warengruppe'] == product_id).astype(int)
        all_features.append(product_col)
        
        # Weather effects
        for weather in weather_features:
            col_name = f'{weather}_product_{product_id}'
            df_with_features[col_name] = df_with_features[product_col] * df_with_features[weather]
            all_features.append(col_name)
            
        # Weather code effects    
        for weather_code in weather_codes:
            col_name = f'{weather_code}_product_{product_id}'
            df_with_features[col_name] = df_with_features[product_col] * df_with_features[weather_code]
            all_features.append(col_name)
            
        # Wind effects
        for wind in wind_features:
            col_name = f'{wind}_product_{product_id}'
            df_with_features[col_name] = df_with_features[product_col] * df_with_features[wind]
            all_features.append(col_name)
        
        # Time effects
        for time in time_features:
            col_name = f'{time}_product_{product_id}'
            df_with_features[col_name] = df_with_features[product_col] * df_with_features[time]
            all_features.append(col_name)
        
        # Event effects
        for event in event_features:
            col_name = f'{event}_product_{product_id}'
            df_with_features[col_name] = df_with_features[product_col] * df_with_features[event]
            all_features.append(col_name)
            
        col_name = f'Temp2_product_{product_id}'
        df_with_features[col_name] = df_with_features[product_col] * df_with_features['Temp2']
        all_features.append(col_name)
    
    return df_with_features, all_features

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['loss'], label='Training')
    ax1.plot(history.history['val_loss'], label='Validation')
    ax1.set_title('Loss Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(history.history['mae'], label='Training')
    ax2.plot(history.history['val_mae'], label='Validation')
    ax2.set_title('Mean Absolute Error Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def prepare_and_predict_umsatz_nn(df):
    """
    Neural network model for predicting turnover with product interactions and polynomial features
    """
    df_with_features, feature_columns = create_product_features(df)
    
    X_train, X_test, y_train, y_test = split_train_validation(df_with_features, feature_columns)
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()
    
    """ model = Sequential([
        Dense(256, activation='relu', input_shape=(len(feature_columns),)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(32, activation='relu'),
        Dense(1)
    ]) """
    
    model = Sequential([
    Dense(128, activation='relu', input_shape=(len(feature_columns),), kernel_regularizer=l2(0.01)),  # Reduced from 256 to 128
    BatchNormalization(),
    Dropout(0.42),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),  # Reduced from 128 to 64
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),  # Reduced from 64 to 32
    Dropout(0.12),
    Dense(1)  # Output layer remains the same
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.000085),
                 loss='mse',
                 metrics=['mae'])
    
    # Using the new callbacks
    callbacks = create_callbacks()
    
    history = model.fit(X_train_scaled, y_train_scaled,
                       epochs=50,
                       batch_size=32,
                       validation_data=(X_test_scaled, y_test_scaled), 
                       callbacks=callbacks,
                       verbose=1)
    
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_unscaled = y_test.values.reshape(-1, 1)
    
    r2 = r2_score(y_test_unscaled, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred))
    
    product_metrics = {}
    for product_id in range(1, 7):
        mask = X_test[f'is_product_{product_id}'] == 1
        if mask.any():
            product_r2 = r2_score(y_test_unscaled[mask], y_pred[mask])
            product_rmse = np.sqrt(mean_squared_error(y_test_unscaled[mask], y_pred[mask]))
            product_metrics[f"Product {product_id}"] = {'R2': product_r2, 'RMSE': product_rmse}
    
    return model, r2, rmse, product_metrics, scaler_X, scaler_y, history, feature_columns

def main():
    # Load and merge data
    df_merged = merge_datasets()
    df_featured = prepare_features(df_merged)
    df_cleaned = handle_missing_values(df_featured)
    
    # Train model
    model, r2, rmse, product_metrics, scaler_X, scaler_y, history, feature_columns = prepare_and_predict_umsatz_nn(df_cleaned)
    
    # Print results
    print(f"\nNeural Network Overall Performance:")
    print(f"R-squared score: {r2:.3f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    
    print("\nNeural Network Performance by Product Category:")
    for category, metrics in product_metrics.items():
        print(f"\n{category}:")
        print(f"R-squared: {metrics['R2']:.3f}")
        print(f"RMSE: {metrics['RMSE']:.2f}")
    
    # Plot training history
    plot_history(history)

if __name__ == "__main__":
    main()