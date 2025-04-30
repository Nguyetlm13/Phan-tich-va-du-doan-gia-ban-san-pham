import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Đọc dữ liệu
df = pd.read_csv('cleaned_shopee_data.csv')

# Tính toán số lượng từng loại
category_counts = df['new_category_code'].value_counts().sort_index()

# Vẽ biểu đồ
plt.figure(figsize=(12, 6))
category_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Số lượng từng mã số (Category Code)', fontsize=14)
plt.xlabel('Mã số (Category Code)', fontsize=12)
plt.ylabel('Số lượng', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()