import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'cleaned_shopee_data.csv'
data = pd.read_csv(file_path)

# Lọc dữ liệu hợp lệ
valid_data = data[['category1', 'favorite']].dropna(subset=['category1', 'favorite'])

# Vẽ biểu đồ boxplot
plt.figure(figsize=(15, 8))
sns.boxplot(
    x='category1',
    y='favorite',
    data=valid_data,
    palette='coolwarm'
)

# Thêm tiêu đề và nhãn
plt.title("Biểu đồ tương quan giữa category1 và favorite")
plt.xlabel("Danh mục sản phẩm (category1)")
plt.ylabel("Số lượng yêu thích (favorite)")
plt.xticks(rotation=45)  # Xoay tên danh mục cho dễ nhìn
plt.show()
