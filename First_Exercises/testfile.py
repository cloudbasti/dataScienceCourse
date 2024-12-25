import pandas as pd
import matplotlib.pyplot as plt

# Sample data: average turnover and confidence intervals for each weekday
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
average_turnover = [200, 250, 300, 280, 320]  # Replace with your calculated averages
confidence_intervals = [(180, 220), (230, 270), (290, 310), (260, 300), (300, 340)]  # Replace with your calculated CIs

# Calculate the lower and upper bounds of the confidence intervals
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
plt.tight_layout()
plt.show()
