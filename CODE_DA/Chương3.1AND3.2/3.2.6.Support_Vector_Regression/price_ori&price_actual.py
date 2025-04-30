# Import thư viện
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load dữ liệu
data = pd.read_csv('cleaned_shopee_data.csv')

# Split dữ liệu thành các feature và target
X = data[['price_ori']]  # Feature
y = data['price_actual']  # Target

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo mô hình SVR
svr_model = SVR(kernel='rbf', C=100, gamma='scale')  # Sử dụng kernel RBF
svr_model.fit(X_train, y_train)  # Huấn luyện SVR

# Dự đoán trên tập dữ liệu kiểm tra
y_pred = svr_model.predict(X_test)

# Tính các chỉ số đánh giá hiệu quả
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# In kết quả
print(f"Mean Squared Error: \n{mse}")
print(f"R² Score: \n{r2}")

# Tạo khoảng smooth (mượt) cho price_ori
price_ori_smooth = pd.DataFrame({
    'price_ori': np.linspace(X['price_ori'].min(), X['price_ori'].max(), 500)
})  # Chuyển numpy array thành DataFrame

# Dự đoán cho khoảng smooth
price_actual_smooth = svr_model.predict(price_ori_smooth)

# Vẽ biểu đồ scatter
plt.figure(figsize=(12, 6))
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label="Actual Data")  # Dữ liệu thực tế
plt.scatter(X_test, y_pred, color='red', alpha=0.5, label="SVR Predictions")  # Dự đoán dữ liệu
plt.plot(price_ori_smooth['price_ori'], price_actual_smooth, color='orange', linewidth=2, label="SVR Regression Line")  # Đường hồi quy mượt

# Thêm thông tin vào biểu đồ
plt.title("Scatter Plot SVR - price_ori vs price_actual", fontsize=16)
plt.xlabel('price_ori', fontsize=12)
plt.ylabel('price_actual', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.show()
