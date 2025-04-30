import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'cleaned_shopee_data.csv'
data = pd.read_csv(file_path)


# Convert 'timestamp' column to datetime format
data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')

# Extract the month from the timestamp column
data['month'] = data['timestamp'].dt.month

# Count occurrences of each month
month_counts = data['month'].value_counts().sort_index()

# Plot a pie chart for the distribution of months
plt.figure(figsize=(8, 8))
plt.pie(month_counts, labels=month_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
plt.title('Distribution of Records by Month', fontsize=16)
plt.show()
