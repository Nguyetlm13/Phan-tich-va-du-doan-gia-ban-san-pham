import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Đọc dữ liệu từ file Excel
data = pd.read_csv('cleaned_shopee_data.csv')

# Tính số lượt yêu thích trung bình theo category1
avg_favorite = data.groupby('category1')['favorite'].mean().sort_values()

# Biểu đồ
plt.figure(figsize=(10, 6))
avg_favorite.plot(kind='bar', color='orange')
plt.title("Số lượt yêu thích trung bình (favorite) theo danh mục (category1)")
plt.xlabel("Danh mục chính (category1)")
plt.ylabel("Số lượt yêu thích trung bình")
plt.xticks(rotation=45)
plt.show()
