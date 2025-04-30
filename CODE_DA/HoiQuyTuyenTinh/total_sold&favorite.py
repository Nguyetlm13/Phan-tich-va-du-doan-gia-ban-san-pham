import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

# Đọc dữ liệu
data = pd.read_csv("cleaned_shopee_data.csv")

# Chuẩn bị dữ liệu cho hồi quy tuyến tính - Mối quan hệ price_ori vs price_actual
X1 = data['price_ori'].values.reshape(-1, 1)  # Biến độc lập
y1 = data['price_actual'].values  # Biến phụ thuộc

# Tạo và huấn luyện mô hình hồi quy tuyến tính
model1 = LinearRegression()
model1.fit(X1, y1)

# Dự đoán
y1_pred = model1.predict(X1)

# Chuẩn bị dữ liệu cho hồi quy tuyến tính - Mối quan hệ total_sold vs favorite
X2 = data['total_sold'].values.reshape(-1, 1)  # Biến độc lập
y2 = data['favorite'].values  # Biến phụ thuộc

# Tạo và huấn luyện mô hình hồi quy tuyến tính
model2 = LinearRegression()
model2.fit(X2, y2)

# Dự đoán
y2_pred = model2.predict(X2)

# Tạo 2 biểu đồ con trong cùng một cửa sổ
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Biểu đồ hồi quy tuyến tính price_ori vs price_actual
sns.scatterplot(x='price_ori', y='price_actual', data=data, ax=axes[0], label="Actual Data", color="blue")
axes[0].plot(data['price_ori'], y1_pred, color='red', label="Regression Line")
axes[0].set_title("Linear Regression: Price_ori vs Price_actual")
axes[0].set_xlabel("Original Price (price_ori)")
axes[0].set_ylabel("Actual Price (price_actual)")
axes[0].legend()
axes[0].grid(True)

# Biểu đồ hồi quy tuyến tính total_sold vs favorite
sns.scatterplot(x='total_sold', y='favorite', data=data, ax=axes[1], label="Actual Data", color="blue")
axes[1].plot(data['total_sold'], y2_pred, color='red', label="Regression Line")
axes[1].set_title("Linear Regression: Total_sold vs Favorite")
axes[1].set_xlabel("Total_sold")
axes[1].set_ylabel("Favorite")
axes[1].legend()
axes[1].grid(True)

# Hiển thị tất cả biểu đồ
plt.tight_layout()
plt.show()
