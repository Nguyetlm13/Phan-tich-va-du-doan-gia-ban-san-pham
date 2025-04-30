import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Đọc dữ liệu từ file CSV
df = pd.read_csv('20240121_shopee_sample_data (1).csv')

# Chuyển đổi các cột liên quan sang kiểu số (cleaning nếu cần)
columns_to_convert = ['total_sold', 'favorite', 'item_rating']
for col in columns_to_convert:
    df[col] = pd.to_numeric(df[col].str.replace(r'[^\d.]', '', regex=True), errors='coerce')

# Hàm áp dụng quy tắc 3-sigma để loại bỏ outliers
def apply_3sigma_rule(data, column):
    mu = data[column].mean()
    sigma = data[column].std()
    lower_bound = mu - 3 * sigma
    upper_bound = mu + 3 * sigma
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
# Hàm vẽ Boxplot theo quy tắc 3-sigma cho từng cặp cột
def plot_3sigma_boxplots_separately(data, column_pairs):
    for col1, col2 in column_pairs:
        # Lọc dữ liệu theo quy tắc 3-sigma
        filtered_data = data.dropna(subset=[col1, col2])
        filtered_data = apply_3sigma_rule(filtered_data, col1)
        filtered_data = apply_3sigma_rule(filtered_data, col2)
        # Tạo một figure mới cho mỗi cặp cột
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=filtered_data[[col1, col2]])
        plt.title(f"3-Sigma Boxplot for {col1} and {col2}", fontsize=14)
        plt.xlabel("Columns")
        plt.ylabel("Values")
        plt.show()
# Danh sách các cặp cột để vẽ biểu đồ
column_pairs = [
    ('price_ori', 'price_actual'),
    ('total_sold', 'favorite'),
    ('item_rating', 'total_sold')
]
# Vẽ biểu đồ Boxplot theo quy tắc 3-sigma
plot_3sigma_boxplots_separately(df, column_pairs)
