import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Load data
data = pd.read_csv("cleaned_shopee_data.csv")

# Chuẩn bị X (Original Price) và Y (Actual Price)
X = np.array(data['total_sold']).reshape(-1, 1)  # price_ori là biến độc lập
Y = np.array(data['favorite']).reshape(-1, 1)  # price_actual là biến phụ thuộc

# Chia dữ liệu thành tập huấn luyện (80%) và tập kiểm tra (20%)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Vẽ biểu đồ scatter
plt.figure(figsize=(10, 6))

# Biểu đồ scatter cho dữ liệu huấn luyện
plt.scatter(X_train, y_train, alpha=0.5, label='Train Data (80%)', color='blue')

# Biểu đồ scatter cho dữ liệu kiểm tra
plt.scatter(X_test, y_test, alpha=0.5, label='Test Data (20%)', color='orange')

# Thêm tiêu đề, nhãn trục và chú thích
plt.title("Relationship Between total_sold and favorite", fontsize=14)
plt.xlabel("total_sold", fontsize=12)
plt.ylabel("favorite", fontsize=12)
plt.legend()
plt.grid(True)

# Hiển thị biểu đồ
plt.show()
