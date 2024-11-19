import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load your data
data = pd.read_csv("umsatzdaten_gekuerzt.csv")

# add weekdays to dataset (not needed in fact)

data['Datum'] = pd.to_datetime(data['Datum'])

data['weekday'] = data['Datum'].dt.day_name()

# Assuming your DataFrame has a 'turnover' column and a 'weekday' column
# Convert the 'weekday' column to categorical if it's not already
data['weekday'] = pd.Categorical(data['weekday'], categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ordered=True)

# Initialize lists to store results
weekdays = data['weekday'].cat.categories
average_turnover = []
confidence_intervals = []

# Loop through each weekday to calculate average turnover and confidence intervals
for day in weekdays:
    daily_data = data[data['weekday'] == day]
    mean_turnover = daily_data["Umsatz"].mean()
    std_dev = daily_data['Umsatz'].std()
    n = daily_data['Umsatz'].count()
    
    # Calculate the standard error of the mean
    sem = std_dev / np.sqrt(n)
    
    # Calculate the critical value for a 95% confidence interval
    alpha = 0.05
    critical_value = stats.norm.ppf(1 - alpha/2)
    
    # Calculate the margin of error and confidence intervals
    margin_of_error = critical_value * sem
    ci_lower = mean_turnover - margin_of_error
    ci_upper = mean_turnover + margin_of_error
    
    # Append results to lists
    average_turnover.append(mean_turnover)
    confidence_intervals.append((ci_lower, ci_upper))

# Prepare data for plotting
lower_bounds = [ci[0] for ci in confidence_intervals]
upper_bounds = [ci[1] for ci in confidence_intervals]

# Create the bar chart
plt.bar(weekdays, average_turnover, yerr=[np.array(average_turnover) - np.array(lower_bounds), 
                                            np.array(upper_bounds) - np.array(average_turnover)],
        capsize=5, color='skyblue', alpha=0.7)

# Add labels and title
plt.xlabel('Weekdays')
plt.ylabel('Average Turnover')
plt.title('Average Turnover per Weekday with Confidence Intervals')

# Show the plot
#plt.tight_layout()
plt.show()
