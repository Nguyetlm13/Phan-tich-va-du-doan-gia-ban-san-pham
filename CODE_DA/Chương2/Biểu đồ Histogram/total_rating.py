import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu từ file Excel
df = pd.read_csv('data_cleaned_total_rating.csv')

# Vẽ biểu đồ histogram
plt.hist(df['total_rating'], bins=20)
plt.xlabel('Total Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Total Ratings')
plt.show()