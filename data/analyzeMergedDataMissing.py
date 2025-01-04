import pandas as pd


def analyze_missing_weather_values(df):
    # Weather columns to check
    weather_cols = ['Bewoelkung', 'Temperatur',
                    'Windgeschwindigkeit', 'Wettercode']

    print("Missing Values Analysis for Weather Data:")
    print("-" * 50)

    # Check total rows
    total_rows = len(df)
    print(f"Total rows in dataset: {total_rows}\n")

    for col in weather_cols:
        # Count different types of missing values
        null_count = df[col].isnull().sum()
        na_count = df[col].isna().sum()
        empty_string_count = (df[col] == '').sum(
        ) if df[col].dtype == 'object' else 0

        # Calculate percentage
        missing_percent = (null_count / total_rows) * 100

        print(f"Column: {col}")
        print(f"Data type: {df[col].dtype}")
        print(f"Null values (isnull): {null_count}")
        print(f"NA values (isna): {na_count}")
        print(f"Empty strings: {empty_string_count}")
        print(f"Percentage missing: {missing_percent:.2f}%")

        # Show some example rows with missing values if any exist
        if null_count > 0:
            print("\nExample rows with missing values:")
            print(df[df[col].isnull()][['Datum', col]].head())

        print("-" * 50)


def main():
    # Load the merged test data
    print("Loading merged test data...")
    test_df = pd.read_csv("data/prepared_test_data.csv")

    # Analyze missing values
    analyze_missing_weather_values(test_df)


if __name__ == "__main__":
    main()
