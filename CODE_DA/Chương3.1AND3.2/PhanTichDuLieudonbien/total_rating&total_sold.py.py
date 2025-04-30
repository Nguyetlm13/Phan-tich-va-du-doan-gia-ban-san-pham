import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
# Reload the CSV file
data = pd.read_csv("cleaned_shopee_data.csv")

# Lấy dữ liệu liên quan
x = data['total_rating'].values.reshape(-1, 1)
y = data['total_sold'].values

# Tạo mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['total_rating'], y=data['total_sold'], alpha=0.6, label='Actual Data')
plt.plot(data['total_rating'], y_pred, color='red', label='Regression Line')
plt.title('Linear Regression: Total Rating vs Total Sold', fontsize=14)
plt.xlabel('Total Rating', fontsize=12)
plt.ylabel('Total Sold', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()