import pandas as pd
import numpy as np


def analyze_weathercode(df):
    # Temporal analysis of missing weather codes
    df['month'] = df['Datum'].dt.month
    df['season'] = pd.cut(df['Datum'].dt.month,
                          bins=[0, 3, 6, 9, 12],
                          labels=['Winter', 'Spring', 'Summer', 'Fall'])

    # Check missing patterns by month
    monthly_missing = df.groupby('month')['Wettercode'].apply(
        lambda x: (x.isna().sum() / len(x)) * 100
    ).round(2)

    # Check missing patterns by season
    seasonal_missing = df.groupby('season')['Wettercode'].apply(
        lambda x: (x.isna().sum() / len(x)) * 100
    ).round(2)

    # Check if missing values correlate with other variables
    # Calculate average conditions when weather code is missing vs. present
    conditions_comparison = pd.DataFrame({
        'Missing_Weathercode': df[df['Wettercode'].isna()][['Temperatur', 'Windgeschwindigkeit', 'Bewoelkung']].mean(),
        'Present_Weathercode': df[df['Wettercode'].notna()][['Temperatur', 'Windgeschwindigkeit', 'Bewoelkung']].mean()
    })

    # Get the most common weather codes overall
    common_codes = df['Wettercode'].value_counts().head(10)

    # Analyze weather codes by month
    monthly_weather_codes = {}
    for month in range(1, 13):
        month_data = df[df['month'] == month]
        monthly_weather_codes[month] = month_data['Wettercode'].value_counts().head(
            5)

    return monthly_missing, seasonal_missing, conditions_comparison, common_codes, monthly_weather_codes


if __name__ == "__main__":
    # Read the CSV file
    df = pd.read_csv('data/wetter.csv', parse_dates=['Datum'])

    # Run the analysis
    monthly_missing, seasonal_missing, conditions_comparison, common_codes, monthly_weather_codes = analyze_weathercode(
        df)

    print("\nPercentage of missing weather codes by month:")
    print(monthly_missing)
    print("\nPercentage of missing weather codes by season:")
    print(seasonal_missing)
    print("\nAverage conditions comparison (missing vs. present weather codes):")
    print(conditions_comparison)
    print("\nMost common weather codes overall:")
    print(common_codes)

    # Print monthly analysis
    month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April',
                   5: 'May', 6: 'June', 7: 'July', 8: 'August',
                   9: 'September', 10: 'October', 11: 'November', 12: 'December'}

    print("\nMost common weather codes by month:")
    for month in range(1, 13):
        print(f"\n{month_names[month]}:")
        print(monthly_weather_codes[month])
