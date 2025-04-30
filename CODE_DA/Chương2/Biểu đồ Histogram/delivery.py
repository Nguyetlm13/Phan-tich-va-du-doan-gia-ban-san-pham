import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Đọc file CSV đã xử lý
df = pd.read_csv("data_clean_delivery.csv")

# Vẽ histogram cho cột "delivery"
sns.histplot(data=df, x="delivery")
plt.title("Phân bố địa điểm giao hàng")
plt.xlabel("Địa điểm giao hàng")
plt.ylabel("Số lượng đơn hàng")
plt.show()