import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Đọc dữ liệu từ file Excel
data = pd.read_csv('cleaned_shopee_data.csv')

# Chuyển đổi cột 'timestamp' thành định dạng ngày tháng
data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')

# Thêm cột 'month' để lấy thông tin tháng từ cột 'timestamp'
data['month'] = data['timestamp'].dt.to_period('M')

# Tính giá gốc trung bình theo tháng và loại sản phẩm (category1)
monthly_avg_price = data.groupby(['month', 'category1'])['price_ori'].mean().unstack(fill_value=0)

# Vẽ biểu đồ cột nhóm, biểu diễn giá gốc trung bình theo tháng và loại sản phẩm
plt.figure(figsize=(14, 7))
monthly_avg_price.plot(kind='bar', figsize=(16, 8), colormap='tab20', width=0.8)
plt.title('Giá gốc trung bình của các loại sản phẩm (category1) theo tháng', fontsize=16)
plt.xlabel('Tháng', fontsize=14)
plt.ylabel('Giá gốc trung bình', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Loại sản phẩm (category1)', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
