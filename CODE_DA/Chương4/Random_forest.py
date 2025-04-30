import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Đọc dữ liệu
data = pd.read_csv("cleaned_shopee_data1.csv")

# Chọn các cột đầu vào và đầu ra
X = data[['price_ori', 'price_actual', 'favorite', 'item_rating']]  # Các đặc trưng
y = data['total_sold']  # Biến mục tiêu

# Tách dữ liệu thành tập huấn luyện và kiểm tra (80% huấn luyện, 20% kiểm tra)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo mô hình Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Huấn luyện mô hình
rf_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = rf_model.predict(X_test)

# Đánh giá mô hình bằng MSE và R2 Score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# In kết quả đánh giá
print(f'Mean Squared Error (MSE): {mse}')
print(f'R2 Score: {r2}')

# Vẽ biểu đồ so sánh giá trị thực tế và giá trị dự đoán
plt.figure(figsize=(10,6))

# Biểu đồ 1: Dự đoán so với thực tế
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # Đường chéo
plt.xlabel('Giá trị thực tế')
plt.ylabel('Giá trị dự đoán')
plt.title('Dự đoán vs. Thực tế (Random Forest)')

# Biểu đồ 2: Biểu đồ lỗi
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test - y_pred, color='green')
plt.hlines(y=0, xmin=y_test.min(), xmax=y_test.max(), colors='red', linestyles='--')
plt.xlabel('Giá trị thực tế')
plt.ylabel('Lỗi dự đoán')
plt.title('Lỗi dự đoán (Random Forest)')

plt.tight_layout()
plt.show()
