# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = 'cleaned_shopee_data.csv'
data = pd.read_csv(file_path)

# Extract the features and target
features = data[['price_ori']].dropna()  # Loại bỏ NaN trong features
target = data['price_actual'][features.index].dropna()  # Loại bỏ NaN trong target tương ứng

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initializing and training the Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)

# Making predictions
rf_pred = rf_reg.predict(X_test)

# Calculating performance metrics
rf_mse = mean_squared_error(y_test, rf_pred)
print(f"Mean Squared Error (MSE): \n{rf_mse}")
rf_r2 = r2_score(y_test, rf_pred)
print(f"R² Score: \n{rf_r2}")

# Generate a range of sorted values for the independent variable
price_ori_sorted = np.linspace(features['price_ori'].min(), features['price_ori'].max(), 500).reshape(-1, 1)

# Convert to DataFrame for compatibility with the trained model
price_ori_sorted_df = pd.DataFrame(price_ori_sorted, columns=['price_ori'])

# Predict using the trained model
price_actual_pred = rf_reg.predict(price_ori_sorted_df)

# Visualizing the Random Forest predictions
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Data', alpha=0.5)
plt.scatter(X_test, rf_pred, color='green', label='Random Forest Predictions', alpha=0.5)
# Plot the regression line
plt.plot(price_ori_sorted_df, price_actual_pred, color='orange', label='Random Forest Regression Line', linewidth=2)

# Configuring the plot
plt.title("Random Forest Regression: price_ori vs price_actual", fontsize=14)
plt.xlabel("price_ori", fontsize=12)
plt.ylabel("price_actual", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

rf_mse, rf_r2
