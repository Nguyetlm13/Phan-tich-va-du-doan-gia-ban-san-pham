from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import pandas as pd

# Load data
data = pd.read_csv('cleaned_shopee_data.csv')

# Split data into training and testing sets
X = data[['total_sold']]  # Giữ nguyên là DataFrame
y = data['favorite']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Thiết lập phạm vi các giá trị k cho GridSearchCV
param_grid = {'n_neighbors': range(1, 21)}  # Tìm k từ 1 đến 20

# Train KNN model và sử dụng GridSearchCV để tìm k tối ưu
knn = KNeighborsRegressor(n_neighbors=5)
grid_search = GridSearchCV(knn, param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)  # Huấn luyện trên tập huấn luyện

# Lấy giá trị k tối ưu
best_k = grid_search.best_params_['n_neighbors']
best_model = grid_search.best_estimator_

# Dự đoán trên tập kiểm tra
y_pred = best_model.predict(X_test)  # X_test vẫn là DataFrame

# Tính toán R² và MSE
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# In kết quả
print(f"Giá trị k tối ưu: {best_k}")
print(f"R² trên tập kiểm tra: \n{r2}")
print(f"Mean Squared Error trên tập kiểm tra: \n{mse}")

# Dự đoán với dữ liệu mới (mở rộng)
# Dự đoán trên khoảng smooth từ dữ liệu test
X_smooth = pd.DataFrame({'total_sold': np.linspace(X_test['total_sold'].min(), X_test['total_sold'].max(), 500)})

# Dự đoán với mô hình đã huấn luyện
y_smooth = best_model.predict(X_smooth)

# Vẽ biểu đồ scatter
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label="Actual Data")  # Dữ liệu thực tế
plt.scatter(X_test, y_pred, color='red', alpha=0.5, label="Predicted Data")  # Dữ liệu dự đoán
plt.plot(X_smooth, y_smooth, color='orange', label="Regression Line (KNN)")  # Đường hồi quy KNN

# Thêm thông tin biểu đồ
plt.title('Scatter Plot of KNN Predictions vs Actual', fontsize=16)
plt.xlabel('total_sold (Test Data)', fontsize=14)
plt.ylabel('favorite', fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()
