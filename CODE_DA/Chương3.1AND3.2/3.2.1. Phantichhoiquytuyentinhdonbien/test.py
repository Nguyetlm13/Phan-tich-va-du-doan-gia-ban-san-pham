import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Đọc dữ liệu
data = pd.read_csv("cleaned_shopee_data.csv")

# Lấy toàn bộ dữ liệu của cột 'total_sold' và 'favorite'
price_df = data[['total_sold', 'favorite']]

# Chuẩn bị X và Y
X = np.array(price_df['total_sold']).reshape(-1, 1)  # total_sold là biến độc lập
Y = np.array(price_df['favorite']).reshape(-1, 1)    # favorite là biến phụ thuộc

# Chia dữ liệu thành tập huấn luyện và kiểm tra (80% huấn luyện, 20% kiểm tra)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=1)

# Khởi tạo mô hình hồi quy tuyến tính
model = LinearRegression()

# Huấn luyện mô hình với dữ liệu huấn luyện
model.fit(X_train, y_train)

# Dự đoán giá trị trên tập kiểm tra
y_pred = model.predict(X_test)

# Vẽ biểu đồ phân tán (scatter plot) và đường hồi quy tuyến tính
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, color='blue', alpha=0.5, label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Linear Regression Line')
plt.title("Linear Regression: Total Sold vs Favorite")
plt.xlabel("Total Sold")
plt.ylabel("Favorite")
plt.legend()
plt.grid(True)
plt.show()

# In các hệ số hồi quy và intercept
print(f"Coefficient: {model.coef_}")
print(f"Intercept: {model.intercept_}")
