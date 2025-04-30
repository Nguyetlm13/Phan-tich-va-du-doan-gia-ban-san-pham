import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Đọc dữ liệu
data = pd.read_csv('cleaned_shopee_data.csv')

# Lọc dữ liệu
X = data[['total_sold']].values  # total_sold là biến độc lập
y = data['favorite'].values     # favorite là biến phụ thuộc

# Chia dữ liệu thành tập huấn luyện và kiểm tra (80% huấn luyện, 20% kiểm tra)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo mô hình Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Huấn luyện mô hình
rf_model.fit(X_train, y_train)

# Dự đoán giá trị của tập kiểm tra
y_pred = rf_model.predict(X_test)

# Tính toán các chỉ số đánh giá
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# In các chỉ số đánh giá
print(f'Mean Squared Error (MSE): {mse}')
print(f'R² Score: {r2}')

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual Data')
plt.scatter(X_test, y_pred, color='green', alpha=0.5, label='Random Forest Predictions')

# Vẽ đường hồi quy dựa trên mô hình Random Forest
X_grid = np.arange(min(X.flatten()), max(X.flatten()), 0.1).reshape(-1, 1)  # Tạo dữ liệu mới để vẽ đường mượt mà hơn
y_grid = rf_model.predict(X_grid)
plt.plot(X_grid, y_grid, color='red', label='Random Forest Regression Line')

# Cài đặt tiêu đề và nhãn
plt.title("Random Forest Regression: total_sold vs favorite")
plt.xlabel("Total Sold")
plt.ylabel("Favorite")
plt.legend()
plt.grid(True)
plt.show()
