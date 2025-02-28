
import os
import sys
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.train_split import split_train_validation  # NOQA
from data.TrainingPreparation.data_prep import prepare_features, merge_datasets, handle_missing_values  # NOQA
# from data.data_prep import analyze_weather_codes, analyze_wind_data  # NOQA
from data.WeatherImputation.Final_wetter_imputation import analyze_weather_code_distribution, print_missing_analysis, impute_weather_data  # NOQA


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


def prepare_and_predict_umsatz(df):
    # Scale only Umsatz
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled['Umsatz'] = scaler.fit_transform(
        df['Umsatz'].values.reshape(-1, 1))

    df_with_interactions, feature_columns = create_interaction_features(
        df_scaled)
    X_train, X_test, y_train, y_test = split_train_validation(
        df_with_interactions, feature_columns)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Inverse transform predictions
    y_test_orig = scaler.inverse_transform(y_test.values.reshape(-1, 1))[:, 0]
    y_pred_orig = scaler.inverse_transform(y_pred.reshape(-1, 1))[:, 0]

    r2 = r2_score(y_test_orig, y_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))

    product_metrics = {}
    unique_products = sorted(df_with_interactions['Warengruppe'].unique())
    for product_id in unique_products:
        mask = X_test[f'is_product_{product_id}'] == 1
        if mask.any():
            product_r2 = r2_score(y_test_orig[mask], y_pred_orig[mask])
            product_rmse = np.sqrt(mean_squared_error(
                y_test_orig[mask], y_pred_orig[mask]))
            product_metrics[f"Product {product_id}"] = {
                'R2': product_r2, 'RMSE': product_rmse}

    base_features = [
        'Temperatur', 'Bewoelkung', 'is_holiday', 'is_school_holiday', 'KielerWoche', 'is_nye', 'is_pre_holiday', 'is_weekend_holiday',
        'is_summer_weekend', 'is_peak_summer', 'is_peak_summer_weekend',
        'weekday_Monday', 'weekday_Tuesday', 'weekday_Wednesday', 'weekday_Thursday',
        'weekday_Friday', 'weekday_Saturday', 'weekday_Sunday',
        'season_Winter', 'season_Spring', 'season_Summer', 'season_Autumn',
        'is_Weekend', 'is_Weekday', 'weather_clear', 'weather_no_precip', 'weather_dust_sand', 'weather_fog',
        'weather_drizzle', 'weather_rain', 'weather_snow', 'weather_shower',
        'weather_thunderstorm', 'wind_calm', 'wind_moderate', 'wind_strong', 'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6',
        'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12'
    ]

    product_equations = {}
    scale_factor = scaler.scale_[0]  # Scale factor for Umsatz
    mean_umsatz = scaler.mean_[0]    # Mean of Umsatz

    for product_id in unique_products:
        feature_coefs = {}
        idx = feature_columns.index(f'is_product_{product_id}')
        product_coef = model.coef_[idx] * scale_factor

        for feature in base_features:
            interaction_col = f'int_{feature}_p{product_id}'
            coef = model.coef_[feature_columns.index(
                interaction_col)] * scale_factor
            feature_coefs[feature] = coef

        product_equations[product_id] = {
            'intercept': model.intercept_ * scale_factor + mean_umsatz,
            'coefficients': feature_coefs
        }

    return model, product_equations, r2, rmse, product_metrics, feature_columns


def print_product_equations(product_equations):
    """Print the linear equation for each product in a readable format."""
    print("\nProduct-specific Linear Equations:")
    ordered_features = [
        'Temperatur', 'Bewoelkung', 'is_holiday', 'is_school_holiday', 'KielerWoche', 'is_nye', 'is_pre_holiday', 'is_weekend_holiday',
        'is_summer_weekend', 'is_peak_summer', 'is_peak_summer_weekend',
        'weekday_Monday', 'weekday_Tuesday', 'weekday_Wednesday', 'weekday_Thursday',
        'weekday_Friday', 'weekday_Saturday', 'weekday_Sunday',
        'season_Winter', 'season_Spring', 'season_Summer', 'season_Autumn',
        'is_Weekend', 'is_Weekday', 'weather_clear', 'weather_no_precip', 'weather_dust_sand', 'weather_fog',
        'weather_drizzle', 'weather_rain', 'weather_snow', 'weather_shower',
        'weather_thunderstorm', 'wind_calm', 'wind_moderate', 'wind_strong', 'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6',
        'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12',
    ]

    for product_id, eq in product_equations.items():
        print(f"\nProduct {product_id}:")
        equation = f"Umsatz = {eq['intercept']:.2f}"
        for feature in ordered_features:
            if feature in eq['coefficients']:
                coef = eq['coefficients'][feature]
                sign = '+' if coef >= 0 else '-'
                equation += f" {sign} {abs(coef):.2f}*{feature}"
        print(equation)


def main():
    # Load and merge data
    df_merged = merge_datasets()

    # Analyze weather codes distribution
    #analyze_weather_codes(df_merged)
    #analyze_wind_data(df_merged)

    df_imputed = impute_weather_data(df_merged)
    df_featured = prepare_features(df_imputed)
    df_cleaned = handle_missing_values(df_featured)

    # Train the model and get results
    model, product_equations, r2, rmse, product_metrics, feature_columns = prepare_and_predict_umsatz(
        df_cleaned)

    # Print overall results
    print(f"\nOverall Model Performance:")
    print(f"R-squared score: {r2:.3f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")

    # Print performance by product category
    print("\nModel Performance by Product Category:")
    for category, metrics in product_metrics.items():
        print(f"\n{category}:")
        print(f"R-squared: {metrics['R2']:.3f}")
        print(f"RMSE: {metrics['RMSE']:.2f}")

    # Print product-specific equations
    print_product_equations(product_equations)


if __name__ == "__main__":
    main()
