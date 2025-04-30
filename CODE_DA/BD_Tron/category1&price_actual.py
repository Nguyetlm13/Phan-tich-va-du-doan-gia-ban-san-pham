import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
file_path = 'cleaned_shopee_data.csv'
data = pd.read_csv(file_path)

# Ensure columns are correctly typed
data['price_actual'] = pd.to_numeric(data['price_actual'], errors='coerce')
data['total_sold'] = pd.to_numeric(data['total_sold'], errors='coerce')
data['category1'] = data['category1'].astype(str)

# Group by 'category1' and calculate average price and total_sold
category_stats = data.groupby('category1').agg(
    average_price=('price_actual', 'mean'),
    average_sold=('total_sold', 'mean')
).sort_values(by='average_price', ascending=False)

# Plot the data
fig, ax1 = plt.subplots(figsize=(14, 8))

# Bar plot for average price
ax1.bar(category_stats.index, category_stats['average_price'], color='skyblue', label='Average Price', alpha=0.7)
ax1.set_xlabel('Product Category', fontsize=14)
ax1.set_ylabel('Average Price', color='blue', fontsize=14)
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_xticklabels(category_stats.index, rotation=45, ha='right', fontsize=12)

# Line plot for average total_sold
ax2 = ax1.twinx()
ax2.plot(category_stats.index, category_stats['average_sold'], color='red', marker='o', label='Average Total Sold')
ax2.set_ylabel('Average Total Sold', color='red', fontsize=14)
ax2.tick_params(axis='y', labelcolor='red')

# Titles and grid
plt.title('Average Price and Average Total Sold by Product Category', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.5)
fig.tight_layout()

# Legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.show()
