import pandas as pd

# Đọc dữ liệu từ file CSV
data = pd.read_csv('20240121_shopee_sample_data (1).csv')

# Kiểm tra xem có giá trị âm nào trong cột 'column_name' không
if (data['price_ori'] < 0).any():
    print("Có giá trị âm trong cột 'price_ori'")

# Lấy ra các hàng có giá trị âm trong cột 'column_name'
negative_rows = data[data['price_ori'] < 0]
print(negative_rows)