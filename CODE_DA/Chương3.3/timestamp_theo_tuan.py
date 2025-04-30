import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Đọc dữ liệu từ file CSV
data = pd.read_csv('cleaned_shopee_data1.csv')

# Chuyển đổi cột 'timestamp' thành định dạng thời gian
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Tạo cột 'week' từ cột 'timestamp'
data['week'] = data['timestamp'].dt.isocalendar().week

# Nhóm dữ liệu theo tuần và sản phẩm, tính tổng số lượng bán được
weekly_sales = data.groupby(['week', 'idHash'])['total_sold'].sum().reset_index()

# Tạo các đặc trưng cho mô hình
weekly_sales['previous_week_sales'] = weekly_sales.groupby('idHash')['total_sold'].shift(1)
weekly_sales['week_of_year'] = weekly_sales['week'] % 52

# Loại bỏ các giá trị thiếu
weekly_sales = weekly_sales.dropna()

# Chia dữ liệu thành tập huấn luyện và kiểm tra
train = weekly_sales[weekly_sales['week'] < 40]
test = weekly_sales[weekly_sales['week'] >= 40]

# Xây dựng mô hình Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(train[['previous_week_sales', 'week_of_year']], train['total_sold'])

# Dự đoán trên tập kiểm tra
predictions = model.predict(test[['previous_week_sales', 'week_of_year']])

# Đánh giá mô hình
mae = mean_absolute_error(test['total_sold'], predictions)
print(f'Mean Absolute Error: {mae}')
