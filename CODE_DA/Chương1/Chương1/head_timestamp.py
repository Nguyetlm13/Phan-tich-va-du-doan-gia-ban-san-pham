import pandas as pd

df = pd.read_csv("20240121_shopee_sample_data (1).csv")

print(" Xem 5 dòng dữ liệu đầu tiên của cột timestamp khi chưa được xử lý")
print(df['timestamp'].head(5))


# unique_values = df["timestamp"].unique()
# print("Các giá trị của cột 'timestamp':\n", unique_values)


# unique_values = df["total_sold"].unique()
# print("Các giá trị của cột 'total_sold':\n", unique_values)

df = pd.read_csv("cleaned_shopee_data.csv")

print("\nSau khi được xử lý")
print(df['timestamp'].head(5))
# unique_values = df["timestamp"].unique()
# print("Các giá trị của cột 'timestamp':\n", unique_values)

# unique_values = df["total_sold"].unique()
# print("Các giá trị của cột 'total_sold':\n", unique_values)
