import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Reloading the dataset to work with the new file containing price_ori and price_actual columns.
data = pd.read_csv('cleaned_shopee_data.csv')

# Define X (independent variable) and y (dependent variable)
X = data[['total_sold']].values  # Original Price
y = data['favorite'].values  # Actual Price

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transform features to polynomial features of degree 3
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)


# Train a linear regression model on polynomial features
model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train)

# Predict on the test data
X_test_poly = poly.transform(X_test)
y_pred_poly = model_poly.predict(X_test_poly)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred_poly)
r2 = r2_score(y_test, y_pred_poly)

# Print model parameters and metrics
coefficients = model_poly.coef_
intercept = model_poly.intercept_

coefficients, intercept, mse, r2

# In các chỉ số ra màn hình
print(f"Regression Coefficients: \n{coefficients}")
print(f"Intercept: \n{intercept}")
print(f"Mean Squared Error (MSE): \n{mse}")
print(f"R² Score: \n{r2}")

# Plotting the polynomial regression curve
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.5, label='Actual Data')
sorted_idx = np.argsort(X.flatten())
plt.plot(X[sorted_idx], model_poly.predict(poly.transform(X))[sorted_idx], color='red', label='Polynomial Regression (Degree 3)')
plt.title("Polynomial Regression (Degree 3): total_sold vs. favorite")
plt.xlabel("Số lượt bán (total_sold)")
plt.ylabel("Số lượt thích (favorite)")
plt.legend()
plt.grid(True)
plt.show()


