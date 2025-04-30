import pandas as pd

df = pd.read_csv("20240121_shopee_sample_data (1).csv")

columns = df.columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

print("Các biến phân loại:", categorical_cols)
print("Các biến số:", numerical_cols)