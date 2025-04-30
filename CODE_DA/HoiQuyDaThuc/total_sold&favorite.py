import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

data = pd.read_csv("cleaned_shopee_data.csv")

# Lấy dữ liệu cần thiết
X = data['total_sold'].values.reshape(-1, 1)  # Biến độc lập (giá gốc)
y = data['favorite'].values  # Biến phụ thuộc (giá thực tế)

# Tạo mô hình hồi quy đa thức bậc 2 (Polynomial Regression)
poly = PolynomialFeatures(degree=2)  # Tạo đa thức bậc 2
X_poly = poly.fit_transform(X)  # Biến đổi dữ liệu
model = LinearRegression()
model.fit(X_poly, y)

# Tạo dữ liệu dự đoán
X_range = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
y_pred = model.predict(poly.transform(X_range))

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.6, label='Dữ liệu thực tế')  # Dữ liệu gốc
plt.plot(X_range, y_pred, color='red', label='Hồi quy đa thức bậc 2')  # Đường hồi quy
plt.title('Hồi quy đa thức giữa total_sold và favorite', fontsize=14)
plt.xlabel('total_sold (số lượt bán)', fontsize=12)
plt.ylabel('favorite (Số lượt thích)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()