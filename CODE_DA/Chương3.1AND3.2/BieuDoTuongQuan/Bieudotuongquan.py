import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Đọc dữ liệu
df = pd.read_csv("cleaned_shopee_data.csv")

# Tính ma trận tương quan
correlation_matrix = df[['price_ori', 'item_rating', 'price_actual', 'total_rating', 'total_sold', 'favorite']].corr()

# Vẽ biểu đồ nhiệt độ
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Ma trận tương quan")
plt.show()