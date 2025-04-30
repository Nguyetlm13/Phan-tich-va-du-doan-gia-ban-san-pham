import pandas as pd

df = pd.read_csv("20240121_shopee_sample_data (1).csv")

print(" Xem 5 dòng dữ liệu đầu tiên của cột item_category_detail khi chưa được xử lý")
print(df['item_category_detail'].head(5))