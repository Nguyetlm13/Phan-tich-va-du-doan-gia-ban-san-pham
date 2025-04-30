import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
# Tùy chọn: Tìm k tối ưu bằng Cross-Validation
from sklearn.model_selection import cross_val_score

# Đọc dữ liệu
df = pd.read_csv("cleaned_shopee_data1.csv")

# Giả sử dữ liệu đã có trong một DataFrame `df`
# Chọn các cột cần thiết
X = df[['price_ori', 'price_actual', 'favorite', 'item_rating']]
y = df['total_sold']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Huấn luyện mô hình KNN Regression
knn = KNeighborsRegressor(n_neighbors=5)  # Bắt đầu với k=5
knn.fit(X_train_scaled, y_train)

# Dự đoán trên tập kiểm tra
y_pred = knn.predict(X_test_scaled)

# Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R² Score:", r2)



errors = []
for k in range(1, 20):  # Thử nghiệm k từ 1 đến 20
    knn = KNeighborsRegressor(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, scoring='neg_mean_squared_error', cv=5)
    errors.append(-scores.mean())

optimal_k = np.argmin(errors) + 1
print("Optimal k:", optimal_k)

# Vẽ biểu đồ Dự đoán vs. Thực tế
plt.figure(figsize=(12, 6))

# Biểu đồ Dự đoán vs. Thực tế
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, color='green', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Giá trị thực tế')
plt.ylabel('Giá trị dự đoán')
plt.title('Dự đoán vs. Thực tế (KNN Regression)')

# Biểu đồ Lỗi Dự đoán
plt.subplot(1, 2, 2)
errors = y_pred - y_test
plt.scatter(y_test, errors, color='green', alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Giá trị thực tế')
plt.ylabel('Lỗi dự đoán')
plt.title('Lỗi dự đoán (KNN Regression)')

plt.tight_layout()
plt.show()