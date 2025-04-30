import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Đọc file CSV
df = pd.read_csv("20240121_shopee_sample_data (1).csv")

# xóa cột dữ liệu không mang lại giá trị 
# Xóa cột 'total_rating'
df_cleaned = df.drop(columns=['total_rating'])

# 1. Xóa dữ liệu thiếu
# Bước 1: Điền các giá trị thiếu trong cột 'delivery' bằng chuỗi 'KL City, Kuala Lumpur'
df['delivery'] = df['delivery'].fillna('KL City, Kuala Lumpur')

# Bước 2: Xóa các hàng có ít nhất một giá trị thiếu (các hàng có NaN)
df_cleaned = df.dropna(axis=0, how='any')

# 2. Chuẩn hóa kiểu dữ liệu
# Chuyển đổi cột 'item_rating' thành kiểu số, thay thế các giá trị không hợp lệ bằng 0
df_cleaned['item_rating'] = pd.to_numeric(df_cleaned['item_rating'], errors='coerce').fillna(0)

# Xử lý các cột có chứa chữ 'k' - sửa lại đoạn này để chỉ nhân với 1000
def process_k_values(value):
    if pd.isna(value):
        return value  # Nếu là giá trị thiếu, giữ nguyên
    if 'k' in value:  # Nếu có chữ 'k'
        value = value.replace('k', '')  # Xóa chữ 'k'
        value = float(value) * 1000  # Nhân giá trị với 1000
    return value

# Sửa lại việc xử lý cột 'total_rating' và 'total_sold'
df_cleaned['total_rating'] = df_cleaned['total_rating'].apply(process_k_values).astype(float)
df_cleaned['total_sold'] = df_cleaned['total_sold'].apply(process_k_values).astype(float)

# Xử lý cột 'favorite'
def preprocess_favorite(value):
    if pd.isna(value):
        return value  # Nếu là giá trị thiếu, giữ nguyên
    value = value.replace("Favorite (", "")
    if 'k' in value:
        value = value.replace('k', '')
        value = float(value) * 1000
        value = int(value)
    return value

df_cleaned['favorite'] = df_cleaned['favorite'].apply(preprocess_favorite)
df_cleaned['favorite'] = pd.to_numeric(df_cleaned['favorite'], errors='coerce')
df_cleaned = df_cleaned.dropna(subset=['favorite'])  # Xóa các hàng có NaN trong 'favorite'
df_cleaned['favorite'] = df_cleaned['favorite'].astype(int)

# Làm sạch cột 'timestamp'
df_cleaned['timestamp'] = pd.to_datetime(df_cleaned['timestamp'], unit='ms').dt.date

# 3. Mã hóa dữ liệu: Tách cột 'item_category_detail' thành các cột mới (nếu cột này tồn tại)
if 'item_category_detail' in df_cleaned.columns:
    df_cleaned[['item', 'category1', 'category2', 'category3']] = df_cleaned['item_category_detail'].str.split('|', expand=True)

    # Đếm số lần xuất hiện của mỗi giá trị trong cột 'category1'
    category1_counts = df_cleaned['category1'].value_counts()

    # Tạo một DataFrame chứa các giá trị trong 'category1' và số lần xuất hiện
    category1_with_codes = pd.DataFrame({
        'category1': category1_counts.index,
        'count': category1_counts.values
    }).reset_index(drop=True)

    # Gán số từ 1 đến 24 cho các giá trị của 'category1' dựa trên thứ tự xuất hiện cao nhất đến thấp nhất
    category1_with_codes['new_category_code'] = range(1, len(category1_with_codes) + 1)

    # Tạo một từ điển ánh xạ từ 'category1' sang 'new_category_code'
    category1_to_code_map = dict(zip(category1_with_codes['category1'], category1_with_codes['new_category_code']))

    # Thêm cột 'new_category_code' vào DataFrame chính bằng cách ánh xạ từ 'category1'
    df_cleaned['new_category_code'] = df_cleaned['category1'].map(category1_to_code_map)

# 4. Xử lý giá trị ngoại lai (Outliers) cho các cột `price_ori`, `price_actual`, `total_sold`, `favorite` sử dụng quy tắc 3-sigma
def remove_outliers(df, column):
    # Tính trung bình và độ lệch chuẩn
    mean = df[column].mean()
    std_dev = df[column].std()
    
    # Tính giới hạn dưới và giới hạn trên theo quy tắc 3-sigma
    lower_limit = mean - 3 * std_dev
    upper_limit = mean + 3 * std_dev
    
    # Lọc bỏ các giá trị nằm ngoài phạm vi này
    return df[(df[column] >= lower_limit) & (df[column] <= upper_limit)]

# Áp dụng hàm xử lý giá trị ngoại lai cho các cột cần thiết
df_cleaned = remove_outliers(df_cleaned, 'price_ori')
df_cleaned = remove_outliers(df_cleaned, 'price_actual')
df_cleaned = remove_outliers(df_cleaned, 'total_sold')
df_cleaned = remove_outliers(df_cleaned, 'favorite')

# 5. Lưu dữ liệu đã làm sạch vào file CSV mới
output_path = "cleaned_shopee_data.csv"
df_cleaned.to_csv(output_path, index=False)

print(f"Dữ liệu đã được làm sạch và lưu tại {output_path}")
