import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

data = pd.read_csv("cleaned_shopee_data.csv")
# For demonstration, I'll assume 'price_actual' is another column we derive from `price_ori`.
# Here, I will randomly simulate some values for `price_actual`.
# In a real case, this should be replaced with actual data or calculation.

# Prepare data for regression
X = data['price_ori'].values.reshape(-1, 1)  # Independent variable
y = data['price_actual'].values             # Dependent variable

# Create and fit the regression model
model = LinearRegression()
model.fit(X, y)

# Generate predictions
y_pred = model.predict(X)

# Plot the regression line and data points
plt.figure(figsize=(10, 6))
sns.scatterplot(x='price_ori', y='price_actual', data=data, label="Actual Data", color="blue")
plt.plot(data['price_ori'], y_pred, color='red', label="Regression Line")
plt.title("Linear Regression: Price_ori vs Price_actual")
plt.xlabel("Original Price (price_ori)")
plt.ylabel("Actual Price (price_actual)")
plt.legend()
plt.grid(True)
plt.show()