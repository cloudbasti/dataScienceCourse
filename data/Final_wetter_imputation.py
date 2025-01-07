import pandas as pd
import numpy as np


def print_missing_analysis(df):
    """Print missing values analysis for weather columns"""
    # weather_cols = ['Bewoelkung', 'Temperatur',
    #                'Windgeschwindigkeit', 'Wettercode']
    # print("\nMissing Values Analysis:")
    # print("-" * 50)

    # for col in weather_cols:
    #     missing = df[col].isna().sum()
    #     print(f"{col}: {missing} missing values ({(missing/len(df))*100:.2f}%)")
    pass


def analyze_weather_code_distribution(df, month, title=""):
    """Analyze weather code distribution for a specific month"""
    month_data = df[df['Datum'].dt.month == month]
    distribution = month_data['Wettercode'].value_counts().sort_index()

    # if len(distribution) > 0:
    #     print(f"\n{title} Distribution for Month {month}:")
    #     print("-" * 60)
    #     print(f"Total values: {len(month_data)}")
    #     print(f"Unique codes: {len(distribution)}")
    #     print("\nTop 10 most frequent codes:")
    #     print(distribution.head(10))
    return distribution
    # return pd.Series()


def impute_weather_data(df):
    """
    Impute missing weather values after merge with distribution analysis.
    """
    df_imputed = df.copy()

    # Print initial analysis
    # print("Before imputation:")
    # print_missing_analysis(df_imputed)

    # Store initial weather code distributions
    initial_distributions = {}
    for month in range(1, 13):
        initial_distributions[month] = analyze_weather_code_distribution(
            df_imputed, month, "Initial Weather Code")

    # 1. Temperature imputation using weekly averages
    print("\nStarting temperature imputation...")
    df_imputed['week'] = df_imputed['Datum'].dt.isocalendar().week

    weekly_temp_avg = df_imputed.groupby('week')['Temperatur'].mean()
    # print("\nWeekly temperature averages:")
    # print(weekly_temp_avg.describe())

    for week in df_imputed['week'].unique():
        mask = (df_imputed['week'] == week) & (df_imputed['Temperatur'].isna())
        df_imputed.loc[mask, 'Temperatur'] = weekly_temp_avg[week]
    print("Temperature imputation completed.")

    # 2. Wind speed imputation using median
    print("\nStarting wind speed imputation...")
    wind_median = df_imputed['Windgeschwindigkeit'].median()
    # print(f"Wind speed median value: {wind_median}")
    df_imputed['Windgeschwindigkeit'] = df_imputed['Windgeschwindigkeit'].fillna(
        wind_median)
    print("Wind speed imputation completed.")

    # 3. Cloud cover imputation using median
    print("\nStarting cloud cover imputation...")
    cloud_median = round(df_imputed['Bewoelkung'].median())
    # print(f"Cloud cover median value: {cloud_median}")
    df_imputed['Bewoelkung'] = df_imputed['Bewoelkung'].fillna(cloud_median)
    print("Cloud cover imputation completed.")

    # 4. Weather code imputation using monthly distribution
    print("\nStarting weather code imputation...")
    df_imputed['month'] = df_imputed['Datum'].dt.month

    for month in range(1, 13):
        # Get monthly data
        month_mask = df_imputed['month'] == month
        month_data = df_imputed[month_mask]

        # Calculate probability distribution of weather codes for this month
        code_counts = month_data['Wettercode'].value_counts()
        if len(code_counts) > 0:
            total_valid = code_counts.sum()
            code_probs = code_counts / total_valid

            # Find missing values for this month
            missing_mask = month_mask & df_imputed['Wettercode'].isna()
            n_missing = missing_mask.sum()

            if n_missing > 0:
                replacement_codes = np.random.choice(
                    code_counts.index,
                    size=n_missing,
                    p=code_probs.values
                )
                df_imputed.loc[missing_mask, 'Wettercode'] = replacement_codes
    print("Weather code imputation completed.")

    # Analyze final distributions and compare
    # print("\nComparing Weather Code Distributions Before and After Imputation:")
    # print("=" * 80)

    # for month in range(1, 13):
    #     print(f"\nMonth {month}:")
    #     print("-" * 60)

    #     # Get before and after distributions
    #     before_dist = initial_distributions[month]
    #     after_dist = analyze_weather_code_distribution(
    #         df_imputed, month, "Final Weather Code")

    #     if not before_dist.empty and not after_dist.empty:
    #         # Combine into a DataFrame for side-by-side comparison
    #         comparison = pd.DataFrame({
    #             'Before': before_dist,
    #             'After': after_dist
    #         }).fillna(0).astype(int)

    #         # Add a difference column
    #         comparison['Difference'] = comparison['After'] - \
    #             comparison['Before']

    #         print("\nDistribution Changes:")
    #         print(comparison.sort_values('After', ascending=False).head(10))

    # Remove helper columns
    df_imputed = df_imputed.drop(['week', 'month'], axis=1)

    # Final analysis
    # print("\nAfter imputation:")
    # print_missing_analysis(df_imputed)

    return df_imputed


def main():
    # Load the merged data
    print("Loading merged data...")
    df = pd.read_csv("data/prepared_test_data.csv")
    df['Datum'] = pd.to_datetime(df['Datum'])

    # Perform imputation
    df_imputed = impute_weather_data(df)

    # Save imputed data
    output_path = "data/test_final.csv"
    df_imputed.to_csv(output_path, index=False)
    print(f"\nSaved imputed data to: {output_path}")

    # Print final statistics
    # print("\nFinal data statistics:")
    # print(df_imputed[['Temperatur', 'Windgeschwindigkeit',
    #       'Bewoelkung', 'Wettercode']].describe())


if __name__ == "__main__":
    main()
