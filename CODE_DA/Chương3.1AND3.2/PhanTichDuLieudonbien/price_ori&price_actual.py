import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reload the CSV file
data = pd.read_csv("cleaned_shopee_data.csv")

# Check if the required columns exist for the new regression
required_columns = ['price_actual', 'price_ori']
if all(column in data.columns for column in required_columns):
    # Remove rows with missing values in the specified columns
    filtered_data = data.dropna(subset=required_columns)
    
    # Plot the regression chart
    plt.figure(figsize=(10, 6))
    sns.regplot(
        x='price_actual', 
        y='price_ori', 
        data=filtered_data, 
        scatter_kws={'alpha': 0.6}, 
        line_kws={'color': 'blue'}
    )
    plt.title('Linear Regression: Price Actual vs Price Original', fontsize=16)
    plt.xlabel('Price Actual', fontsize=14)
    plt.ylabel('Price Original', fontsize=14)
    plt.grid(alpha=0.3)
    plt.show()
else:
    f"One or both of the required columns {required_columns} are missing in the dataset."