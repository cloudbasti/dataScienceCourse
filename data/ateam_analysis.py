import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_overall_trends(df):
    daily_turnover = df.groupby('Datum')['Umsatz'].sum().reset_index()
    monthly_turnover = df.groupby([df['Datum'].dt.year, df['Datum'].dt.month])[
        'Umsatz'].mean()

    daily_turnover.to_csv("analysis_results/daily_turnover.csv", index=False)
    monthly_turnover.to_csv("analysis_results/monthly_turnover.csv")
    return daily_turnover, monthly_turnover


def analyze_product_groups(df):
    group_avg = df.groupby('Warengruppe')['Umsatz'].agg(
        ['mean', 'std', 'min', 'max']).round(2)
    season_group = df.groupby(['season', 'Warengruppe'])[
        'Umsatz'].mean().unstack()

    group_avg.to_csv("analysis_results/product_group_stats.csv")
    season_group.to_csv("analysis_results/seasonal_product_performance.csv")
    return group_avg, season_group


def analyze_weather_impact(df):
    weather_impact = pd.DataFrame()
    weather_columns = [col for col in df.columns if col.startswith('weather_')]

    for weather_type in weather_columns:
        avg_turnover = df[df[weather_type] == 1]['Umsatz'].mean()
        weather_impact.loc[weather_type, 'avg_turnover'] = avg_turnover

    temp_correlation = df.groupby('Warengruppe')[
        'Umsatz'].corr(df['Temperatur']).round(3)

    weather_impact.to_csv("analysis_results/weather_impact.csv")
    temp_correlation.to_csv("analysis_results/temperature_correlation.csv")
    return weather_impact, temp_correlation


def analyze_temporal_patterns(df):
    weekday_patterns = df.groupby(['Wochentag', 'Warengruppe'])[
        'Umsatz'].mean().unstack()

    holiday_impact = pd.DataFrame({
        'Regular_Days': df[df['is_holiday'] == 0]['Umsatz'].mean(),
        'Holidays': df[df['is_holiday'] == 1]['Umsatz'].mean(),
        'Pre_Holidays': df[df['is_pre_holiday'] == 1]['Umsatz'].mean()
    }, index=['Average_Turnover'])

    weekday_patterns.to_csv("analysis_results/weekday_patterns.csv")
    holiday_impact.to_csv("analysis_results/holiday_impact.csv")
    return weekday_patterns, holiday_impact


def analyze_special_events(df):
    kiwo_impact = pd.DataFrame({
        'Regular_Days': df[df['KielerWoche'] == 0]['Umsatz'].mean(),
        'Kieler_Woche': df[df['KielerWoche'] == 1]['Umsatz'].mean()
    }, index=['Average_Turnover'])

    kiwo_impact.to_csv("analysis_results/kiwo_impact.csv")
    return kiwo_impact


def analyze_salary_days(df):
    df = df.copy()
    df['day_of_month'] = df['Datum'].dt.day
    df['is_last_day'] = df['Datum'].dt.is_month_end.astype(int)
    df['is_first_day'] = (df['day_of_month'] == 1).astype(int)

    # Define periods
    df['period_type'] = 'Regular'
    df.loc[df['day_of_month'] <= 3, 'period_type'] = 'Start of Month'
    df.loc[df['day_of_month'] >= 28, 'period_type'] = 'End of Month'

    salary_period_stats = pd.DataFrame({
        'First_Day': df[df['is_first_day'] == 1].groupby('Datum')['Umsatz'].sum().mean(),
        'Start_of_Month': df[df['period_type'] == 'Start of Month'].groupby('Datum')['Umsatz'].sum().mean(),
        'End_of_Month': df[df['period_type'] == 'End of Month'].groupby('Datum')['Umsatz'].sum().mean(),
        'Last_Day': df[df['is_last_day'] == 1].groupby('Datum')['Umsatz'].sum().mean(),
        'Regular_Period': df[df['period_type'] == 'Regular'].groupby('Datum')['Umsatz'].sum().mean()
    }, index=['Average_Daily_Turnover'])

    plt.figure(figsize=(14, 6))
    bars = plt.bar(['First Day', 'Start of Month', 'End of Month', 'Last Day', 'Regular Period'],
                   [salary_period_stats.loc['Average_Daily_Turnover', 'First_Day'],
                   salary_period_stats.loc['Average_Daily_Turnover',
                                           'Start_of_Month'],
                   salary_period_stats.loc['Average_Daily_Turnover',
                                           'End_of_Month'],
                   salary_period_stats.loc['Average_Daily_Turnover', 'Last_Day'],
                   salary_period_stats.loc['Average_Daily_Turnover', 'Regular_Period']])

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'€{height:,.0f}',
                 ha='center', va='bottom')

    plt.title('Average Daily Turnover: Salary Period vs Regular Period')
    plt.ylabel('Average Daily Turnover (€)')
    plt.tight_layout()
    plt.savefig('analysis_results/salary_period_comparison.png')
    plt.close()

    salary_period_stats.to_csv('analysis_results/salary_period_stats.csv')

    # Add daily pattern analysis
    daily_pattern = df.groupby('day_of_month')['Umsatz'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    plt.plot(daily_pattern['day_of_month'], daily_pattern['Umsatz'])
    plt.title('Average Turnover by Day of Month')
    plt.xlabel('Day of Month')
    plt.ylabel('Average Turnover (€)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('analysis_results/daily_pattern.png')
    plt.close()

    return salary_period_stats


def analyze_christmas_eve(df):
    df = df.copy()
    df['is_christmas_eve'] = ((df['Datum'].dt.month == 12) &
                              (df['Datum'].dt.day == 24)).astype(int)

    christmas_stats = pd.DataFrame({
        'Christmas_Eve': df[df['is_christmas_eve'] == 1].groupby('Datum')['Umsatz'].sum().mean(),
        'Regular_Days': df[df['is_christmas_eve'] == 0].groupby('Datum')['Umsatz'].sum().mean()
    }, index=['Average_Daily_Turnover'])

    plt.figure(figsize=(10, 6))
    bars = plt.bar(['Christmas Eve', 'Regular Days'],
                   [christmas_stats.loc['Average_Daily_Turnover', 'Christmas_Eve'],
                   christmas_stats.loc['Average_Daily_Turnover', 'Regular_Days']])

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'€{height:,.0f}',
                 ha='center', va='bottom')

    plt.title('Average Daily Turnover: Christmas Eve vs Regular Days')
    plt.ylabel('Average Daily Turnover (€)')
    plt.tight_layout()
    plt.savefig('analysis_results/christmas_eve_comparison.png')
    plt.close()

    product_christmas = df.pivot_table(
        values='Umsatz',
        index='Warengruppe',
        columns='is_christmas_eve',
        aggfunc='mean'
    ).round(2)
    product_christmas.columns = ['Regular_Days', 'Christmas_Eve']
    product_christmas['Relative_Difference'] = (
        (product_christmas['Christmas_Eve'] /
         product_christmas['Regular_Days'] - 1) * 100
    ).round(2)

    christmas_stats.to_csv('analysis_results/christmas_eve_stats.csv')
    product_christmas.to_csv(
        'analysis_results/christmas_eve_product_stats.csv')

    # Add product group comparison plot
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(product_christmas)),
            product_christmas['Relative_Difference'])
    plt.xticks(range(len(product_christmas)),
               product_christmas.index, rotation=45)
    plt.title('Relative Difference in Turnover on Christmas Eve by Product Group')
    plt.ylabel('Relative Difference (%)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('analysis_results/christmas_eve_products.png')
    plt.close()

    return christmas_stats, product_christmas


def create_summary_report(df):
    summary = []
    summary.append("=== Overall Turnover Statistics ===")
    summary.append(f"Total turnover: {df['Umsatz'].sum():,.2f}")
    summary.append(f"Average daily turnover: {
                   df.groupby('Datum')['Umsatz'].sum().mean():,.2f}")
    summary.append(f"Number of unique product groups: {
                   df['Warengruppe'].nunique()}")

    top_days = df.groupby('Datum')['Umsatz'].sum(
    ).sort_values(ascending=False).head()
    summary.append("\n=== Top 5 Days by Turnover ===")
    for date, turnover in top_days.items():
        summary.append(f"{date.strftime('%Y-%m-%d')}: {turnover:,.2f}")

    with open("analysis_results/summary_report.txt", "w") as f:
        f.write("\n".join(summary))


def main():
    os.makedirs("analysis_results", exist_ok=True)

    df = pd.read_csv("data/processed_data.csv")
    df['Datum'] = pd.to_datetime(df['Datum'])

    print("Analyzing overall trends...")
    analyze_overall_trends(df)

    print("Analyzing product groups...")
    analyze_product_groups(df)

    print("Analyzing weather impact...")
    analyze_weather_impact(df)

    print("Analyzing temporal patterns...")
    analyze_temporal_patterns(df)

    print("Analyzing special events...")
    analyze_special_events(df)

    print("Analyzing salary period patterns...")
    analyze_salary_days(df)

    print("Analyzing Christmas Eve patterns...")
    analyze_christmas_eve(df)

    print("Creating summary report...")
    create_summary_report(df)

    print("Analysis complete! Results saved in 'analysis_results' directory.")


if __name__ == "__main__":
    main()
