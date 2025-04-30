# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

file_path = 'cleaned_shopee_data.csv'
data = pd.read_csv(file_path)

# Extract relevant columns
features = data[['total_sold']]
target = data['favorite']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create and train the Random Forest model
rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Calculate MSE and R²
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mse, r2
# Calculating performance metrics
print(f"Mean Squared Error (MSE): \n{mse}")
print(f"R² Score: \n{r2}")

# Plot the comparison of actual vs predicted values
plt.figure(figsize=(12, 6))

# Scatter plot for actual vs predicted
plt.scatter(y_test, y_pred, alpha=0.6, color='b', label='Predicted vs Actual')

# Line y=x for reference
x = np.linspace(y_test.min(), y_test.max(), 100)
plt.plot(x, x, color='r', linestyle='--', label='Ideal Prediction (y=x)')

# Labels and legend
plt.title("Random Forest Regression: Actual vs Predicted")
plt.xlabel("Actual Favorite Count")
plt.ylabel("Predicted Favorite Count")
plt.legend()
plt.grid(True)
plt.show()