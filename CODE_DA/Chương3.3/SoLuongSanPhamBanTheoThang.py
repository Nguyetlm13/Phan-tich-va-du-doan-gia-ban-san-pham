import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Đọc dữ liệu từ file Excel
data = pd.read_csv('cleaned_shopee_data1.csv')

# Chuyển đổi cột 'timestamp' thành định dạng ngày tháng
data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')

# Thêm cột 'month' để lấy thông tin tháng từ cột 'timestamp'
data['month'] = data['timestamp'].dt.to_period('M')

# Tính tổng số lượng bán theo tháng và loại sản phẩm (category1)
monthly_sales = data.groupby(['month', 'category1'])['total_sold'].sum().unstack(fill_value=0)

# Tính tổng số lượng bán theo từng loại sản phẩm trên toàn bộ dữ liệu
category_totals = monthly_sales.sum(axis=0)

# Cập nhật chú thích với số lượng từng loại sản phẩm
labels = [f"{category} ({int(total)})" for category, total in category_totals.items()]

# Vẽ lại biểu đồ với chú thích mới
plt.figure(figsize=(12, 6))
ax = monthly_sales.plot(kind='bar', stacked=True, figsize=(14, 7), colormap='tab20')
plt.title('Tổng số lượng bán theo loại sản phẩm (category1) mỗi tháng', fontsize=16)
plt.xlabel('Tháng', fontsize=14)
plt.ylabel('Tổng số lượng bán', fontsize=14)
plt.legend(labels=labels, title='Loại sản phẩm (category1) và tổng số lượng', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()