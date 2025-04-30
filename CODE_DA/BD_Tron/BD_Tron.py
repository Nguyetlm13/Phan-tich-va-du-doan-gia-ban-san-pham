import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'cleaned_shopee_data.csv'
data = pd.read_csv(file_path)

# Map `new_category_code` to `category1` for descriptive labels
category_labels = data.groupby('new_category_code')['category1'].first()
category_counts = data['new_category_code'].value_counts()

# Calculate percentage for each category
category_percentages = (category_counts / category_counts.sum() * 100).round(1)

# Combine `new_category_code`, `category1`, and percentages for the legend
detailed_labels = [
    f"{code}: {category_labels.get(code, 'Unknown')} ({percentage}%)"
    for code, percentage in zip(category_counts.index, category_percentages)
]

# Create a larger pie chart with a legend
plt.figure(figsize=(12, 10))

# Plot the pie chart without labels inside the chart
category_counts.plot.pie(
    autopct='',
    startangle=90,
    cmap='tab20',
    legend=False,
    textprops={'fontsize': 12}
)

# Add a legend outside the chart with detailed labels
plt.legend(
    labels=detailed_labels,
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    fontsize=10,
    title="Categories"
)

# Add a title
plt.title("Distribution of New Category Codes with Detailed Legend", fontsize=14)

# Display the chart
plt.ylabel('')  # Remove the default y-axis label
plt.show()
