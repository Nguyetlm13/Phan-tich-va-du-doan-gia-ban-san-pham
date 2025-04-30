import pandas as pd
import matplotlib.pyplot as plt

# Đọc file CSV
df = pd.read_csv("data_cleaned_w_date.csv")

# Vẽ biểu đồ histogram
plt.hist(df['w_date'], bins=30)  # Bạn có thể điều chỉnh số lượng bins tùy ý
plt.xlabel("Ngày")
plt.ylabel("Số lượng")
plt.title("Phân phối các ngày trong cột w_date")

plt.xticks(rotation=45)  # Xoay nhãn trục x

plt.show()