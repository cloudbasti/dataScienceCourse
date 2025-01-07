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


def analyze_product_4(df):
    """Detailed analysis for Product 4 including temporal patterns and external factors"""
    df_p4 = df[df['Warengruppe'] == 4].copy()

    # Basic statistics
    basic_stats = df_p4['Umsatz'].describe().round(2)
    basic_stats.to_csv('analysis_results/product4_basic_stats.csv')

    # Monthly analysis
    monthly_stats = df_p4.groupby(df_p4['Datum'].dt.month)['Umsatz'].agg([
        'mean', 'std', 'count', 'median', 'min', 'max'
    ]).round(2)

    plt.figure(figsize=(12, 6))
    monthly_means = monthly_stats['mean']
    bars = plt.bar(range(1, 13), monthly_means)
    plt.title('Product 4: Average Monthly Turnover')
    plt.xlabel('Month')
    plt.ylabel('Average Turnover (€)')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'€{height:.2f}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('analysis_results/product4_monthly.png')
    plt.close()

    # Weekday analysis with detailed statistics
    dow_stats = df_p4.groupby('Wochentag')['Umsatz'].agg([
        'mean', 'std', 'count', 'median', 'min', 'max'
    ]).round(2)

    plt.figure(figsize=(12, 6))
    weekdays = ['Monday', 'Tuesday', 'Wednesday',
                'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_means = [dow_stats.loc[day, 'mean']
                     if day in dow_stats.index else 0 for day in weekdays]

    bars = plt.bar(weekdays, weekday_means)
    plt.title('Product 4: Average Turnover by Weekday')
    plt.xlabel('Weekday')
    plt.ylabel('Average Turnover (€)')
    plt.xticks(rotation=45)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'€{height:.2f}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('analysis_results/product4_weekday.png')
    plt.close()

    # Time series decomposition
    daily_sales = df_p4.groupby('Datum')['Umsatz'].sum().reset_index()
    daily_sales.set_index('Datum', inplace=True)

    # Calculate rolling statistics
    rolling_mean = daily_sales['Umsatz'].rolling(window=7).mean()
    rolling_std = daily_sales['Umsatz'].rolling(window=7).std()

    plt.figure(figsize=(15, 7))
    plt.plot(daily_sales.index,
             daily_sales['Umsatz'], label='Daily Sales', alpha=0.5)
    plt.plot(daily_sales.index, rolling_mean,
             label='7-day Moving Average', linewidth=2)
    plt.fill_between(daily_sales.index,
                     rolling_mean - rolling_std,
                     rolling_mean + rolling_std,
                     alpha=0.2, label='±1 STD Range')
    plt.title('Product 4: Daily Sales with Rolling Statistics')
    plt.xlabel('Date')
    plt.ylabel('Turnover (€)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('analysis_results/product4_time_series.png')
    plt.close()

    # Weather impact analysis
    weather_impact = pd.DataFrame()
    weather_cols = [col for col in df_p4.columns if col.startswith('weather_')]

    for weather in weather_cols:
        weather_stats = df_p4[df_p4[weather] == 1]['Umsatz'].agg([
            'mean', 'std', 'count', 'median'
        ]).round(2)
        weather_impact = pd.concat(
            [weather_impact, weather_stats.to_frame(weather)], axis=1)

    # Temperature analysis
    temp_correlation = df_p4['Umsatz'].corr(df_p4['Temperatur'])

    # Create temperature bins and analyze sales
    df_p4['temp_bin'] = pd.cut(df_p4['Temperatur'],
                               bins=[-float('inf'), 5, 10, 15,
                                     20, 25, float('inf')],
                               labels=['<5°C', '5-10°C', '10-15°C', '15-20°C', '20-25°C', '>25°C'])

    temp_stats = df_p4.groupby('temp_bin', observed=True)['Umsatz'].agg([
        'mean', 'std', 'count', 'median'
    ]).round(2)

    # Outlier analysis
    Q1 = df_p4['Umsatz'].quantile(0.25)
    Q3 = df_p4['Umsatz'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df_p4[(df_p4['Umsatz'] < lower_bound) |
                     (df_p4['Umsatz'] > upper_bound)].copy()

    # Calculate outlier dates and their characteristics
    outlier_details = pd.DataFrame({
        'date': outliers['Datum'],
        'turnover': outliers['Umsatz'],
        'weekday': outliers['Wochentag'],
        'temperature': outliers['Temperatur'],
        'is_holiday': outliers['is_holiday']
    }).sort_values('turnover', ascending=False)

    # Holiday analysis
    holiday_impact = pd.DataFrame({
        'Regular_Days': df_p4[df_p4['is_holiday'] == 0]['Umsatz'].mean(),
        'Holidays': df_p4[df_p4['is_holiday'] == 1]['Umsatz'].mean(),
        'Pre_Holidays': df_p4[df_p4['is_pre_holiday'] == 1]['Umsatz'].mean()
    }, index=['Average_Turnover'])

    # Save all results
    monthly_stats.to_csv('analysis_results/product4_monthly_stats.csv')
    dow_stats.to_csv('analysis_results/product4_weekday_stats.csv')
    weather_impact.to_csv('analysis_results/product4_weather_impact.csv')
    temp_stats.to_csv('analysis_results/product4_temperature_stats.csv')
    outlier_details.to_csv('analysis_results/product4_outliers.csv')
    holiday_impact.to_csv('analysis_results/product4_holiday_impact.csv')

    # Create a summary dictionary
    summary_stats = {
        'basic_stats': basic_stats,
        'monthly_stats': monthly_stats,
        'dow_stats': dow_stats,
        'weather_impact': weather_impact,
        'temp_correlation': temp_correlation,
        'temp_stats': temp_stats,
        'outliers_count': len(outliers),
        'outlier_percentage': round((len(outliers) / len(df_p4) * 100), 2),
        'holiday_impact': holiday_impact
    }

    # Create a text summary
    with open('analysis_results/product4_summary.txt', 'w') as f:
        f.write("=== Product 4 Analysis Summary ===\n\n")
        f.write(f"Total days analyzed: {len(df_p4)}\n")
        f.write(f"Average daily turnover: €{basic_stats['mean']:.2f}\n")
        f.write(f"Median daily turnover: €{basic_stats['50%']:.2f}\n")
        f.write(f"Standard deviation: €{basic_stats['std']:.2f}\n")
        f.write(f"Temperature correlation: {temp_correlation:.3f}\n")
        f.write(f"Number of outliers: {len(outliers)} ({
                (len(outliers)/len(df_p4)*100):.1f}%)\n")
        f.write(f"\nBest performing month: {
                monthly_stats['mean'].idxmax()} (€{monthly_stats['mean'].max():.2f})\n")
        f.write(f"Worst performing month: {
                monthly_stats['mean'].idxmin()} (€{monthly_stats['mean'].min():.2f})\n")

    return summary_stats

def analyze_product_6(df):
    """Detailed analysis for Product 6 with seasonal focus"""
    df_p6 = df[df['Warengruppe'] == 6].copy()

    # Add seasonal period flag
    df_p6['is_seasonal_period'] = df_p6['Datum'].dt.month.isin(
        [1, 10, 11, 12]).astype(int)

    # Analyze weekday patterns within seasonal period
    seasonal_weekday_stats = df_p6[df_p6['is_seasonal_period'] == 1].groupby('Wochentag')['Umsatz'].agg([
        'mean', 'std', 'count'
    ]).round(2)

    # Create weekday turnover visualization
    plt.figure(figsize=(12, 6))
    weekdays = ['Monday', 'Tuesday', 'Wednesday',
                'Thursday', 'Friday', 'Saturday', 'Sunday']
    seasonal_means = [seasonal_weekday_stats.loc[day, 'mean']
                      if day in seasonal_weekday_stats.index else 0 for day in weekdays]

    bars = plt.bar(weekdays, seasonal_means)
    plt.title('Product 6: Average Turnover by Weekday (Seasonal Period Only)')
    plt.xlabel('Weekday')
    plt.ylabel('Average Turnover (€)')
    plt.xticks(rotation=45)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'€{height:.2f}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('analysis_results/product6_seasonal_weekday.png')
    plt.close()

    # Calculate and save seasonal statistics
    seasonal_stats = pd.DataFrame({
        'in_season_avg': df_p6[df_p6['is_seasonal_period'] == 1]['Umsatz'].mean(),
        'out_of_season_avg': df_p6[df_p6['is_seasonal_period'] == 0]['Umsatz'].mean(),
        'in_season_days': df_p6[df_p6['is_seasonal_period'] == 1]['Umsatz'].count(),
        'out_of_season_days': df_p6[df_p6['is_seasonal_period'] == 0]['Umsatz'].count()
    }, index=['Product 6'])

    seasonal_stats.to_csv('analysis_results/product6_seasonal_stats.csv')
    seasonal_weekday_stats.to_csv(
        'analysis_results/product6_seasonal_weekday_stats.csv')
    df_p6 = df[df['Warengruppe'] == 6].copy()

    # Outlier Analysis
    Q1 = df_p6['Umsatz'].quantile(0.25)
    Q3 = df_p6['Umsatz'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df_p6[(df_p6['Umsatz'] < lower_bound) |
                     (df_p6['Umsatz'] > upper_bound)].copy()

    plt.figure(figsize=(12, 6))
    plt.scatter(df_p6['Datum'], df_p6['Umsatz'], alpha=0.5, label='Normal')
    plt.scatter(outliers['Datum'], outliers['Umsatz'],
                color='red', alpha=0.7, label='Outliers')
    plt.title('Product 6: Turnover Distribution with Outliers')
    plt.xlabel('Date')
    plt.ylabel('Turnover (€)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('analysis_results/product6_outliers.png')
    plt.close()

    outliers_stats = pd.DataFrame({
        'total_points': len(df_p6),
        'outliers': len(outliers),
        'outlier_percentage': (len(outliers) / len(df_p6)) * 100,
        'mean_turnover': df_p6['Umsatz'].mean(),
        'outlier_mean_turnover': outliers['Umsatz'].mean()
    }, index=['Product 6'])

    monthly_stats = df_p6.groupby(df_p6['Datum'].dt.month)['Umsatz'].agg([
        'mean', 'std', 'count'
    ]).round(2)

    plt.figure(figsize=(12, 6))
    monthly_stats['mean'].plot(kind='bar')
    plt.title('Product 6: Average Monthly Turnover')
    plt.xlabel('Month')
    plt.ylabel('Average Turnover (€)')
    plt.tight_layout()
    plt.savefig('analysis_results/product6_monthly.png')
    plt.close()

    weather_impact = pd.DataFrame()
    weather_cols = [col for col in df_p6.columns if col.startswith('weather_')]

    for weather in weather_cols:
        avg_turnover = df_p6[df_p6[weather] == 1]['Umsatz'].mean()
        weather_impact.loc[weather, 'avg_turnover'] = avg_turnover

    plt.figure(figsize=(12, 6))
    weather_impact['avg_turnover'].plot(kind='bar')
    plt.title('Product 6: Average Turnover by Weather Condition')
    plt.xlabel('Weather Condition')
    plt.ylabel('Average Turnover (€)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('analysis_results/product6_weather.png')
    plt.close()

    dow_stats = df_p6.groupby('Wochentag')['Umsatz'].agg([
        'mean', 'std', 'count'
    ]).round(2)

    plt.figure(figsize=(10, 6))
    dow_stats['mean'].plot(kind='bar')
    plt.title('Product 6: Average Turnover by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Average Turnover (€)')
    plt.tight_layout()
    plt.savefig('analysis_results/product6_dow.png')
    plt.close()

    temp_corr = df_p6['Umsatz'].corr(df_p6['Temperatur'])
    plt.figure(figsize=(10, 6))
    plt.scatter(df_p6['Temperatur'], df_p6['Umsatz'], alpha=0.5)
    plt.title(
        f'Product 6: Turnover vs Temperature (correlation: {temp_corr:.2f})')
    plt.xlabel('Temperature')
    plt.ylabel('Turnover (€)')
    plt.tight_layout()
    plt.savefig('analysis_results/product6_temperature.png')
    plt.close()

    monthly_stats.to_csv('analysis_results/product6_monthly_stats.csv')
    outliers_stats.to_csv('analysis_results/product6_outlier_stats.csv')
    weather_impact.to_csv('analysis_results/product6_weather_impact.csv')
    dow_stats.to_csv('analysis_results/product6_dow_stats.csv')

    return {
        'outliers_stats': outliers_stats,
        'monthly_stats': monthly_stats,
        'weather_impact': weather_impact,
        'dow_stats': dow_stats,
        'temp_correlation': temp_corr
    }
    

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

    print("Analyzing Product 6 (Seasonal Bread)...")
    analyze_product_6(df)
    
    analyze_product_4(df)

    print("Creating summary report...")
    create_summary_report(df)

    print("Analysis complete! Results saved in 'analysis_results' directory.")


if __name__ == "__main__":
    main()
