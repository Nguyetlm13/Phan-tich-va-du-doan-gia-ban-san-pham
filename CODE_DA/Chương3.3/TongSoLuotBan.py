import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Đọc dữ liệu từ file Excel
data = pd.read_csv('cleaned_shopee_data.csv')

# Tính tổng số lượng bán theo category1
total_sold = data.groupby('category1')['total_sold'].sum().sort_values()

# Biểu đồ
plt.figure(figsize=(10, 6))
total_sold.plot(kind='barh', color='green')
plt.title("Tổng số lượng bán (total_sold) theo danh mục (category1)")
plt.xlabel("Số lượng bán")
plt.ylabel("Danh mục chính (category1)")
plt.show()
