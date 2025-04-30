import matplotlib.pyplot as plt
import pandas as pd

# Đọc file CSV đã xử lý
df = pd.read_csv("cleaned_shopee_data.csv")

# Danh sách các cột cần vẽ biểu đồ histogram
columns_to_plot = ['price_ori', 'delivery', 'price_actual', 'total_sold', 'total_rating', 'w_date', 'timestamp', 'favorite', 'item_rating', 'new_category_code']

# Vẽ histogram cho từng cột
def plot_histograms(data, columns):
    """
    Vẽ biểu đồ histogram cho các cột trong danh sách.

    Parameters:
    - data: DataFrame chứa dữ liệu.
    - columns: Danh sách các cột để vẽ histogram.

    Returns:
    - None: Hiển thị các biểu đồ histogram.
    """
    for col in columns:
        if col in data.columns:
            plt.figure(figsize=(8, 6))
            if col == 'w_date' or col == 'timestamp':
                # Xử lý riêng cho cột w_date và timestamp nếu là ngày tháng
                plt.hist(data[col].dropna(), bins=30, color='skyblue', edgecolor='black')  # Histogram
                plt.title(f'Histogram of {col}', fontsize=14)
                plt.xlabel(col, fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.xticks(rotation=45)  # Xoay nhãn trục x nghiêng 45 độ
            elif col == 'new_category_code':
                # Xử lý cho cột new_category_code để hiện thị giá trị số liên tiếp
                plt.hist(data[col].dropna().astype(int), bins=len(data[col].dropna().unique()), color='skyblue', edgecolor='black')  # Histogram
                plt.title(f'Histogram of {col}', fontsize=14)
                plt.xlabel(col, fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
            else:
                plt.hist(data[col].dropna(), bins=30, color='skyblue', edgecolor='black')  # Histogram
                plt.title(f'Histogram of {col}', fontsize=14)
                plt.xlabel(col, fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()  # Đảm bảo bố cục biểu đồ gọn gàng
            plt.show()
        else:
            print(f"Column '{col}' not found in DataFrame.")

# Gọi hàm để vẽ biểu đồ histogram
plot_histograms(df, columns_to_plot)
