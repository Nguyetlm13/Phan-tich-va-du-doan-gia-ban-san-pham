import pandas as pd

# Đọc dữ liệu từ file CSV
df = pd.read_csv('cleaned_shopee_data.csv')

# Hiển thị 5 dòng đầu tiên của cột 'item_category_detail'
print("Hiển thị 5 dòng đầu tiên của cột 'item_category_detail'")
print(df['item_category_detail'].head())