import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def merge_datasets():
    # Load datasets
    weather = pd.read_csv("data/wetter_imputed.csv")
    turnover = pd.read_csv("data/umsatzdaten_gekuerzt.csv")
    kiwo = pd.read_csv("data/kiwo.csv")
    school_holidays = pd.read_csv("data/school_holidays.csv")
    public_holidays = pd.read_csv("data/bank_holidays.csv")

    # Convert dates
    for df in [weather, turnover, kiwo, school_holidays, public_holidays]:
        df['Datum'] = pd.to_datetime(df['Datum'])

    # Merge all datasets
    df = pd.merge(turnover, weather, on='Datum', how='left')
    df = pd.merge(df, kiwo, on='Datum', how='left')
    df = pd.merge(df, school_holidays, on='Datum', how='left')
    df = pd.merge(df, public_holidays, on='Datum', how='left')

    # Fill NaN values
    for col in ['KielerWoche', 'is_school_holiday', 'is_holiday']:
        df[col] = df[col].fillna(0)

    return df


def prepare_features(df):
    df_prepared = df.copy()

    # Create season feature
    df_prepared['season'] = df_prepared['Datum'].dt.month.map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    })

    # Weekend feature
    df_prepared['is_weekend'] = df_prepared['Datum'].dt.dayofweek.isin(
        [5, 6]).map({True: 'Weekend', False: 'Weekday'})

    # Create target variables for analysis
    df_prepared['is_weekend_holiday'] = ((df_prepared['is_weekend'] == 'Weekend') & (
        df_prepared['is_holiday'] == 1)).astype(int)
    df_prepared['is_peak_summer'] = df_prepared['Datum'].dt.month.isin([
                                                                       7, 8]).astype(int)

    return df_prepared


def create_confidence_plots(df):
    # Set basic style
    plt.rcParams['figure.figsize'] = (15, 6)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

    fig, (ax1, ax2) = plt.subplots(1, 2)

    def get_stats(data):
        mean = np.mean(data)
        std_err = stats.sem(data)
        ci = stats.t.interval(confidence=0.95,
                              df=len(data)-1,
                              loc=mean,
                              scale=std_err)
        return mean, ci[0], ci[1]

    # Weekday vs Weekend Analysis
    categories = ['Weekdays', 'Weekend']
    means, ci_lower, ci_upper = [], [], []

    # For weekdays (Monday-Friday)
    data_weekday = df[df['is_weekend'] == 'Weekday']['Umsatz']
    mean, ci_low, ci_high = get_stats(data_weekday)
    means.append(mean)
    ci_lower.append(mean - ci_low)
    ci_upper.append(ci_high - mean)

    # For weekend
    data_weekend = df[df['is_weekend'] == 'Weekend']['Umsatz']
    mean, ci_low, ci_high = get_stats(data_weekend)
    means.append(mean)
    ci_lower.append(mean - ci_low)
    ci_upper.append(ci_high - mean)

    ax1.bar(categories, means, yerr=[
            ci_lower, ci_upper], capsize=5, color='skyblue')
    ax1.set_title('Sales by Day Type')
    ax1.set_ylabel('Average Sales')
    ax1.tick_params(axis='x', rotation=45)

    # Peak Summer Analysis
    categories = ['Regular Season', 'Peak Summer']
    means, ci_lower, ci_upper = [], [], []

    for val in [0, 1]:
        data = df[df['is_peak_summer'] == val]['Umsatz']
        mean, ci_low, ci_high = get_stats(data)
        means.append(mean)
        ci_lower.append(mean - ci_low)
        ci_upper.append(ci_high - mean)

    ax2.bar(categories, means, yerr=[
            ci_lower, ci_upper], capsize=5, color='lightgreen')
    ax2.set_title('Sales by Season')
    ax2.set_ylabel('Average Sales')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('confidence_intervals.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # Load and prepare data
    df = merge_datasets()
    df_prepared = prepare_features(df)

    # Create plots
    create_confidence_plots(df_prepared)
    print("Plots have been saved as 'confidence_intervals.png'")


if __name__ == "__main__":
    main()
