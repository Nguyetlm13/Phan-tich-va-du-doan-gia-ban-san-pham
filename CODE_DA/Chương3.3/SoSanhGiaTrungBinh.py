import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Đọc dữ liệu từ file Excel
data = pd.read_csv('cleaned_shopee_data.csv')

# Tính giá trung bình theo category1
avg_price = data.groupby('category1')['price_actual'].mean().sort_values()

# Biểu đồ
plt.figure(figsize=(10, 6))
avg_price.plot(kind='bar', color='skyblue')
plt.title("Giá trung bình (price_actual) theo danh mục (category1)")
plt.xlabel("Danh mục chính (category1)")
plt.ylabel("Giá bán trung bình")
plt.xticks(rotation=45)
plt.show()
