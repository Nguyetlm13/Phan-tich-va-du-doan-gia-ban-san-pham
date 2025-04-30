from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load the dataset
# Đọc file CSV
file_path = 'cleaned_shopee_data.csv'  # Đảm bảo đường dẫn đúng
data = pd.read_csv(file_path)


# Chọn các cột liên quan
features = ['price_ori', 'favorite', 'price_actual']
target = 'total_sold'

# Lọc và loại bỏ giá trị thiếu (nếu có)
data_filtered = data[features + [target]].dropna()

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X = data_filtered[features]
y = data_filtered[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình Random Forest Regressor
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Lấy một cây đơn lẻ từ mô hình Random Forest
single_tree = model.estimators_[0]

# Vẽ lại cây với độ sâu tối đa là 3 để đơn giản hóa
plt.figure(figsize=(20, 10))
plot_tree(single_tree, feature_names=features, filled=True, rounded=True, fontsize=10, max_depth=3)
plt.title("Simplified Decision Tree from Random Forest (Max Depth = 3)", fontsize=16)
plt.show()
