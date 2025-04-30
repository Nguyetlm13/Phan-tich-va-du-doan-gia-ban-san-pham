import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file Excel
df = pd.read_csv('cleaned_item_rating.csv')

# Vẽ biểu đồ histogram
plt.hist(df['item_rating'], bins=10)  # Tạo 10 bins (khối)
plt.xlabel('Rating')
plt.ylabel('Số lượng')
plt.title('Biểu đồ phân phối tần suất của rating')
plt.show()