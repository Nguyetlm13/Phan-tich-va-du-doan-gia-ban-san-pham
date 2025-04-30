import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Đọc dữ liệu
data = pd.read_csv('cleaned_shopee_data.csv')

# Lọc dữ liệu
data_filtered = data[['total_sold', 'favorite']].dropna()

# Biến độc lập và phụ thuộc
X = data_filtered['total_sold'].values.reshape(-1, 1)  # total_sold
y = data_filtered['favorite'].values                  # favorite

# Tạo đặc trưng đa thức bậc 3
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

# Hồi quy tuyến tính trên dữ liệu đa thức
model = LinearRegression()
model.fit(X_poly, y)

# Dự đoán giá trị
y_pred = model.predict(X_poly)

# Tính toán các chỉ số đánh giá
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f'MSE: {mse}, R2: {r2}')

# Vẽ biểu đồ
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', alpha=0.5, label='Actual Data')
plt.plot(X, y_pred, color='red', label='Polynomial Regression Line (Degree 3)')
plt.title('Polynomial Regression (Degree 3): Total Sold vs Favorite')
plt.xlabel('Total Sold')
plt.ylabel('Favorite')
plt.legend()
plt.grid(True)
plt.show()
