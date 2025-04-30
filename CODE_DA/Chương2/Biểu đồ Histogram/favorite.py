import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file Excel
df = pd.read_csv('data_cleaned_favorite.csv')

plt.hist(df['favorite_count'], bins=20)  # Bạn có thể thay đổi số lượng bins tùy ý
plt.xlabel('Số lượng favorite')
plt.ylabel('Tần suất')
plt.title('Biểu đồ phân phối số lượng favorite')
plt.show()