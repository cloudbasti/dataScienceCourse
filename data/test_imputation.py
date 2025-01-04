import pandas as pd
import numpy as np


def analyze_missing_weather_values(df):
    weather_cols = ['Bewoelkung', 'Temperatur',
                    'Windgeschwindigkeit', 'Wettercode']

    print("Missing Values Analysis for Weather Data:")
    print("-" * 50)

    total_rows = len(df)
    print(f"Total rows in dataset: {total_rows}\n")

    for col in weather_cols:
        null_count = df[col].isnull().sum()
        missing_percent = (null_count / total_rows) * 100

        print(f"Column: {col}")
        print(f"Null values: {null_count}")
        print(f"Percentage missing: {missing_percent:.2f}%")
        print("-" * 50)


def impute_weather_values(df, historical_weather):
    # Convert dates to datetime and extract month for both dataframes
    df['Datum'] = pd.to_datetime(df['Datum'])
    df['month'] = df['Datum'].dt.month

    historical_weather['Datum'] = pd.to_datetime(historical_weather['Datum'])
    historical_weather['month'] = historical_weather['Datum'].dt.month

    # Temperature: monthly averages
    monthly_temp_avg = historical_weather.groupby('month')['Temperatur'].mean()
    print("\nMonthly temperature averages:")
    print(monthly_temp_avg)
    df['Temperatur'] = df.apply(lambda row: monthly_temp_avg[row['month']]
                                if pd.isna(row['Temperatur']) else row['Temperatur'], axis=1)

    # Cloud cover: monthly median
    monthly_cloud_median = historical_weather.groupby('month')[
        'Bewoelkung'].median()
    print("\nMonthly cloud cover medians:")
    print(monthly_cloud_median)
    df['Bewoelkung'] = df.apply(lambda row: monthly_cloud_median[row['month']]
                                if pd.isna(row['Bewoelkung']) else row['Bewoelkung'], axis=1)

    # Wind speed: monthly median
    monthly_wind_median = historical_weather.groupby(
        'month')['Windgeschwindigkeit'].median()
    print("\nMonthly wind speed medians:")
    print(monthly_wind_median)
    df['Windgeschwindigkeit'] = df.apply(lambda row: monthly_wind_median[row['month']]
                                         if pd.isna(row['Windgeschwindigkeit']) else row['Windgeschwindigkeit'], axis=1)

    # Weather code: monthly mode
    monthly_weather_mode = historical_weather.groupby(
        'month')['Wettercode'].agg(lambda x: x.mode()[0])
    print("\nMonthly most common weather codes:")
    print(monthly_weather_mode)
    df['Wettercode'] = df.apply(lambda row: monthly_weather_mode[row['month']]
                                if pd.isna(row['Wettercode']) else row['Wettercode'], axis=1)

    # Remove helper column
    df = df.drop('month', axis=1)

    return df


def main():
    # Load the prepared test data and historical weather data
    print("Loading datasets...")
    test_df = pd.read_csv("data/prepared_test_data.csv")
    historical_weather = pd.read_csv("data/wetter.csv")

    # Analyze missing values before imputation
    print("\nBefore imputation:")
    analyze_missing_weather_values(test_df)

    # Perform imputation
    print("\nPerforming imputation...")
    imputed_df = impute_weather_values(test_df, historical_weather)

    # Analyze missing values after imputation
    print("\nAfter imputation:")
    analyze_missing_weather_values(imputed_df)

    # Save the final imputed dataset
    output_path = "data/test_final.csv"
    imputed_df.to_csv(output_path, index=False)
    print(f"\nSaved final imputed test data to: {output_path}")


if __name__ == "__main__":
    main()
