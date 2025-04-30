import pandas as pd

# đọc file CSV
df= pd.read_csv('cleaned_shopee_data.csv')

# kiểm tra tổng số giá trị null trong mỗi cột
print(df.isnull().sum())

# kiểm tra có bất kì giá trị null nào trong DataFrame không
print(df.isnull().values.any())