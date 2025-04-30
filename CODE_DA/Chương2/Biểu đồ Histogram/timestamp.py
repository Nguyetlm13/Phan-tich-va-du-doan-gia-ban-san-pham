import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu
df = pd.read_csv('data_cleaned_timestamp.csv')


# Vẽ biểu đồ histogram
plt.hist(df['timestamp'], bins=20)  # 20 bins là một ví dụ, bạn có thể thay đổi
plt.xlabel('Thời gian')
plt.ylabel('Tần suất')
plt.title('Biểu đồ phân bố thời gian')

plt.xticks(rotation=45)

plt.show()