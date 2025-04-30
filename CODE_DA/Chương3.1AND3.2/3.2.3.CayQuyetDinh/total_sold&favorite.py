from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('cleaned_shopee_data.csv')

# Chọn cột cần thiết
features = data[['total_sold']]
target = data['favorite']

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Sắp xếp X_test_df theo cột 'price_ori'
X_test_sorted = X_test.sort_values(by='total_sold')
y_test_sorted = y_test.loc[X_test_sorted.index]

# Khởi tạo và huấn luyện mô hình cây quyết định
tree_model = DecisionTreeRegressor(max_depth=5)  # Thử nghiệm với các giá trị khác nhau
tree_model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = tree_model.predict(X_test_sorted)


# In các chỉ số ra màn hình
mse = mean_squared_error(y_test_sorted, y_pred)
print(f"Mean Squared Error (MSE): \n{mse}")
r2 = r2_score(y_test_sorted, y_pred)
print(f"R² Score: \n{r2}")

# Vẽ biểu đồ scatter và đường dự đoán
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color="blue", label="Dữ liệu thực tế", alpha=0.5)
plt.plot(X_test_sorted, y_pred, color="green", label="Decision Tree Model", linewidth=2)

# Cấu hình biểu đồ
plt.title("Decision Tree Regression: Mối quan hệ giữa total_sold và favorite", fontsize=14)
plt.xlabel("total_sold", fontsize=12)
plt.ylabel("favorite", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
