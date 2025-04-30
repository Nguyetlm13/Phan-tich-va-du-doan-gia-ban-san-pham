import matplotlib.pyplot as plt
import pandas as pd

# Đọc file CSV đã xử lý
df = pd.read_csv("20240121_shopee_sample_data (1).csv")

# Danh sách các cột cần vẽ biểu đồ histogram
columns_to_plot = ['delivery', 'price_ori', 'price_actual', 'total_sold', 'w_date', 'timestamp', 'favorite', 'item_rating']


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
            if col == 'w_date':
                # Xử lý riêng cho cột w_date nếu là ngày tháng
                plt.hist(data[col].dropna(), bins=30, color='skyblue', edgecolor='black')  # Histogram
                plt.title(f'Histogram of {col}', fontsize=14)
                plt.xlabel(col, fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.xticks(rotation=45)  # Xoay nhãn trục x nghiêng 45 độ
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
