from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
data = pd.read_csv("cleaned_shopee_data.csv")

# Extract relevant columns and remove missing values
df = data[['price_ori', 'price_actual']].dropna()

# Tạo và vẽ đường hồi quy tuyến tính
X = df[['price_ori']]  # Biến độc lập (giá gốc)
y = df['price_actual']  # Biến phụ thuộc (giá thực tế)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
lin_df = LinearRegression()
lin_df.fit(X_train, y_train)

# Predict on the test set
y_pred = lin_df.predict(X_test)

# Plot the training data, test data, and regression line
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Test Data')
plt.scatter(X_train, y_train, color='green', alpha=0.5, label='Training Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.title('Linear Regression: price_ori vs price_actual (Train/Test Split)')
plt.xlabel('price_ori')
plt.ylabel('price_actual')
plt.legend()
plt.grid(True)
plt.show()

# Lấy hệ số hồi quy (slope) và hệ số chặn (intercept)
coefficients = lin_df.coef_ #Trả về danh sách các hệ số hồi quy tương ứng với các biến đầu vào
intercept = lin_df.intercept_ #Trả về hệ số chặn của mô hình hồi quy tuyến tính.

# Calculate MAE, MSE, and R2 score
# Tính sai số MAE
mae = mean_absolute_error(y_test, y_pred) #Tính sai số tuyệt đối trung bình giữa giá trị thực tế và giá trị dự đoán.
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# In các chỉ số ra màn hình
print(f"Mean Absolute Error (MAE): \n{mae}")
print(f"Regression Coefficients: \n{coefficients}")
print(f"Intercept: \n{intercept}")
print(f"Mean Squared Error (MSE): \n{mse}")
print(f"R² Score: \n{r2}")