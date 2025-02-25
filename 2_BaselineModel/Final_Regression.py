import os
import sys
# from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from data.train_split import split_train_validation  # NOQA
from data.data_prep import prepare_features, merge_datasets, handle_missing_values  # NOQA
#from data.data_prep import analyze_weather_codes, analyze_wind_data  # NOQA


def create_interaction_features(df):
    df_with_interactions = df.copy()

    weekday_columns = [
        col for col in df_with_interactions.columns if col.startswith('weekday_')]
    month_columns = [
        col for col in df_with_interactions.columns if col.startswith('month_')]
    season_columns = [
        col for col in df_with_interactions.columns if col.startswith('season_')]
    weekend_columns = [col for col in df_with_interactions.columns if col.startswith(
        'is_Weekend') or col.startswith('is_Weekday')]
    weather_category_columns = [
        col for col in df_with_interactions.columns if col.startswith('weather_')]
    wind_category_columns = [
        col for col in df_with_interactions.columns if col.startswith('wind_')]

    base_features = ['Temperatur', 'Bewoelkung', 'is_holiday', 'is_school_holiday',
                     'KielerWoche', 'is_nye', 'is_pre_holiday', 'is_weekend_holiday', 'is_summer_weekend', 'is_peak_summer', 'is_peak_summer_weekend'] + weekday_columns + month_columns + season_columns + weekend_columns + weather_category_columns + wind_category_columns

    # Add a print statement to check features
    print("Features being used:", base_features)

    product_dummies = pd.get_dummies(
        df_with_interactions['Warengruppe'], prefix='is_product')
    product_features = list(product_dummies.columns)

    new_columns = {}
    for col in product_features:
        new_columns[col] = product_dummies[col]

    for product_id in df_with_interactions['Warengruppe'].unique():
        product_col = f'is_product_{product_id}'
        for feature in base_features:
            interaction_col = f'int_{feature}_p{product_id}'
            new_columns[interaction_col] = product_dummies[product_col] * \
                df_with_interactions[feature]

    interactions_df = pd.DataFrame(new_columns)
    result_df = pd.concat([
        df_with_interactions[['Datum', 'Umsatz', 'Warengruppe']],
        interactions_df
    ], axis=1)

    all_features = product_features + \
        [col for col in interactions_df.columns if col.startswith('int_')]
    return result_df, all_features


def predict_test_data(model, scaler, feature_columns, test_df):
    """Make predictions on test data and format for submission."""
    # Create features for test data
    test_with_interactions, _ = create_interaction_features(test_df)
    X_test = test_with_interactions[feature_columns]
    # Predict and inverse transform
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))

    # Create submission dataframe
    predictions_df = pd.DataFrame({
        'id': test_df['id'],
        'Umsatz': y_pred.flatten()
    })

    return predictions_df

# ===== MODIFIED prepare_and_predict_umsatz FUNCTION =====


def prepare_and_predict_umsatz(df):
    # Scale only Umsatz
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled['Umsatz'] = scaler.fit_transform(
        df['Umsatz'].values.reshape(-1, 1))

    df_with_interactions, feature_columns = create_interaction_features(
        df_scaled)

    # Train on full dataset instead of splitting
    model = LinearRegression()
    model.fit(df_with_interactions[feature_columns], df_scaled['Umsatz'])

    return model, scaler, feature_columns  # Modified return values

# ===== MODIFIED main FUNCTION =====


def main():
    # Load and merge data
    df_merged = merge_datasets()

    # Analyze weather codes distribution
    #analyze_weather_codes(df_merged)
    #analyze_wind_data(df_merged)

    # Prepare features
    df_featured = prepare_features(df_merged)

    # Handle missing values
    df_cleaned = handle_missing_values(df_featured)

    # Train the model on full dataset
    model, scaler, feature_columns = prepare_and_predict_umsatz(df_cleaned)

    # Load and predict on test data
    test_df = pd.read_csv("data/test_final.csv")
    predictions_df = predict_test_data(model, scaler, feature_columns, test_df)

    # Save predictions
    predictions_df.to_csv("data/submission.csv", index=False)
    print("\nPredictions saved to submission.csv")


if __name__ == "__main__":
    main()
