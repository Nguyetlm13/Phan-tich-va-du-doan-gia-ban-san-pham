import pandas as pd

# Đọc dữ liệu từ file CSV (thay thế 'your_data.csv' bằng đường dẫn file của bạn)
df = pd.read_csv("20240121_shopee_sample_data (1).csv")

# Đếm số lượng giá trị không null trong mỗi cột
for column in df.columns:
    print(f"Số lượng giá trị trong cột {column}: {df[column].count()}")