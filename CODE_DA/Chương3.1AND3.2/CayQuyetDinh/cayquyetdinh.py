# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
import numpy as np

# Load the dataset
file_path = 'cleaned_shopee_data.csv'  # Đảm bảo đường dẫn đúng
data = pd.read_csv(file_path)

# Chọn các cột liên quan
features = ['price_ori', 'favorite', 'price_actual']
target = 'total_sold'


# Chia dữ liệu thành tập huấn luyện và kiểm tra
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tạo và huấn luyện mô hình cây quyết định
tree_model = DecisionTreeRegressor(max_depth=4, random_state=42)  # Giới hạn độ sâu để dễ vẽ
tree_model.fit(X_train, y_train)

# Vẽ cây quyết định
plt.figure(figsize=(20, 10))
plot_tree(tree_model, feature_names=features, 
          filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree for Predicting Total Sold", fontsize=16)
plt.show()
