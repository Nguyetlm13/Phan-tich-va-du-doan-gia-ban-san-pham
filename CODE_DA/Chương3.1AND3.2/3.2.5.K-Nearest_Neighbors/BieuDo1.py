from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import pandas as pd

# Load data
data = pd.read_csv('cleaned_shopee_data.csv')

# Split data into training and testing sets
X = data[['price_ori']]
y = data['price_actual']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Thiết lập phạm vi các giá trị k cho GridSearchCV
param_grid = {'n_neighbors': range(1, 21)}  # Tìm k từ 1 đến 20

# Step 2: Train a KNN Regressor model
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

# Sử dụng GridSearchCV để tìm mô hình KNN tối ưu
grid_search = GridSearchCV(knn, param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)  # Huấn luyện trên tập huấn luyện

# Lấy giá trị k tối ưu
best_k = grid_search.best_params_['n_neighbors']
# Dự đoán trên tập kiểm tra với mô hình tốt nhất
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
  # Sử dụng best_model để dự đoán

# Tính toán R² và MSE trên tập kiểm tra
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
# In kết quả
print(f"Giá trị k tối ưu: {best_k}")
print(f"R² trên tập kiểm tra: {r2}")
print(f"Mean Squared Error trên tập kiểm tra: {mse}")

# Sắp xếp X_test và y_test đồng thời để giữ đúng thứ tự
X_test_sorted, y_test_sorted = zip(*sorted(zip(X_test.values.flatten(), y_test.values.flatten())))

# Chuyển đổi thành mảng numpy
X_test_sorted = np.array(X_test_sorted).reshape(-1, 1)
y_test_sorted = np.array(y_test_sorted)

# Làm mượt bằng spline interpolation (cho trực quan hóa tốt hơn)
X_smooth = np.linspace(X_test_sorted.min(), X_test_sorted.max(), 500)  # Tạo các điểm mượt hơn
X_smooth = X_smooth.reshape(-1, 1)  # Reshape thành mảng 2D
y_smooth = best_model.predict(X_smooth)  # Dự đoán cho các điểm mượt

# Vẽ biểu đồ scatter
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label="Actual Data")  # Dữ liệu thực
plt.scatter(X_test, y_pred, color='red', alpha=0.5, label="Predicted Data")  # Dữ liệu dự đoán
plt.plot(X_smooth, y_smooth, color='orange', label="Regression Line (KNN)")  # Đường hồi quy KNN
plt.title('Scatter Plot of KNN Predictions vs Actual', fontsize=16)
plt.xlabel('price_ori (Test Data)', fontsize=14)
plt.ylabel('price_actual', fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()

