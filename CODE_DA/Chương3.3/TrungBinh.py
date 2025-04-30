import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Đọc dữ liệu từ file Excel
data = pd.read_csv('cleaned_shopee_data.csv')

# Đầu tiên, kiểm tra định dạng của cột 'timestamp' để đảm bảo dữ liệu đúng chuẩn datetime
data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')

# Lọc dữ liệu cho tháng 4 và tháng 5 năm 2023
data_april = data[data['timestamp'].dt.strftime('%Y-%m') == '2023-04']
data_may = data[data['timestamp'].dt.strftime('%Y-%m') == '2023-05']

# Nhóm dữ liệu theo 'category1' và tính tổng 'favorite' và 'total_sold' cho từng tháng
summary_april = data_april.groupby('category1')[['favorite', 'total_sold']].sum().reset_index()
summary_may = data_may.groupby('category1')[['favorite', 'total_sold']].sum().reset_index()

# Gắn cột tháng để dễ phân biệt
summary_april['month'] = 'April'
summary_may['month'] = 'May'

# Gộp dữ liệu hai tháng để dễ so sánh
summary_combined = pd.concat([summary_april, summary_may], ignore_index=True)

# Chuẩn bị dữ liệu cho biểu đồ cột nhóm
categories = summary_combined['category1'].unique()
months = ['April', 'May']

# Lượt yêu thích (favorite) theo tháng
favorites_april = summary_combined[summary_combined['month'] == 'April']['favorite']
favorites_may = summary_combined[summary_combined['month'] == 'May']['favorite']

# Doanh số (total_sold) theo tháng
sold_april = summary_combined[summary_combined['month'] == 'April']['total_sold']
sold_may = summary_combined[summary_combined['month'] == 'May']['total_sold']

# Thiết lập vị trí cho các cột
x = np.arange(len(categories))  # Vị trí của các nhóm sản phẩm
width = 0.35  # Độ rộng của mỗi cột
# Tạo lại biểu đồ với màu sắc cùng tông cho từng chỉ số
fig, ax = plt.subplots(figsize=(15, 8))

# Vẽ cột cho lượt yêu thích (favorite) với cùng tông màu xanh
ax.bar(x - width/2, favorites_april, width, label='Favorite (April)', color='blue', alpha=0.7)
ax.bar(x + width/2, favorites_may, width, label='Favorite (May)', color='blue', alpha=0.4)

# Vẽ cột cho doanh số (total_sold) với cùng tông màu cam
ax.bar(x - width/2, sold_april, width, bottom=favorites_april, label='Sold (April)', color='red', alpha=0.7)
ax.bar(x + width/2, sold_may, width, bottom=favorites_may, label='Sold (May)', color='red', alpha=0.4)

# Cài đặt nhãn và tiêu đề
ax.set_xlabel('Category1', fontsize=12)
ax.set_ylabel('Values (Favorites + Sold)', fontsize=12)
ax.set_title('Favorites and Sold by Category1 in April and May (Unified Colors)', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=10)
ax.legend()

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()