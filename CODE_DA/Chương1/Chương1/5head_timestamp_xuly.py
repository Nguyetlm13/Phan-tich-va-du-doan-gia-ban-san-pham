import pandas as pd

df = pd.read_csv("cleaned_shopee_data.csv")

print(" Xem 5 dòng dữ liệu đầu tiên của cột timestamp khi chưa được xử lý")
print(df['timestamp'].head(5))