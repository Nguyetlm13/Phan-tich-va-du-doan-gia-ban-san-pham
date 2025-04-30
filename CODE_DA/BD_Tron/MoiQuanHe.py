import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'cleaned_shopee_data.csv'
data = pd.read_csv(file_path)

# Lọc dữ liệu hợp lệ
valid_data = data[['price_actual', 'favorite', 'total_sold']].dropna(subset=['price_actual', 'favorite', 'total_sold'])

# Tạo Scatter Plot với color encoding
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='price_actual',
    y='favorite',
    hue='total_sold',
    size='total_sold',
    data=valid_data,
    palette='viridis',
    alpha=0.7
)

# Thêm tiêu đề và nhãn
plt.title("Mối quan hệ giữa Giá thực tế, Số lượng yêu thích và Tổng số bán")
plt.xlabel("Giá thực tế")
plt.ylabel("Yêu thích (Favorite)")
plt.legend(title="Total Sold", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
