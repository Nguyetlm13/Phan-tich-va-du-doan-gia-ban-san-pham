import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu từ file Excel
df = pd.read_csv('cleaned_shopee_data.csv')

# Vẽ biểu đồ histogram
plt.hist(df['price_actual'], bins=30, edgecolor='black')
plt.xlabel("Price_actual")
plt.ylabel("Frequency")
plt.title("Phân phối giá gốc của sản phẩm (sau khi làm sạch)")
plt.ylim(0, 500)  # Giới hạn trục y từ 0 đến 500
plt.show()