from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Đọc tệp dữ liệu
file_path = 'cleaned_shopee_data.csv'
data = pd.read_csv(file_path)

# Chọn các cột cần thiết
relevant_data = data[['price_ori', 'item_rating', 'favorite', 'total_sold', 'price_actual']]

# Tách tập dữ liệu thành biến đầu vào (X) và đầu ra (y)
X = relevant_data[['price_ori', 'item_rating', 'favorite', 'total_sold']]
y = relevant_data['price_actual']

# Chia tập dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tạo biến hồi quy đa thức bậc 2
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Huấn luyện mô hình
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test_poly)

# Tính toán lỗi
mse = mean_squared_error(y_test, y_pred)
print(f"Lỗi bình phương trung bình (MSE): {mse:.2f}")

# Nhập giá trị từ bàn phím
price_ori_input = float(input("Nhập giá trị price_ori (Giá gốc của sản phẩm): "))
total_sold_input = float(input("Nhập giá trị total_sold (Số lượt bán): "))
favorite_input = int(input("Nhập giá trị favorite (Số lượt thích): "))
item_rating_input = float(input("Nhập giá trị item_rating (Đánh giá từ 1.0 đến 5.0): "))

# Tạo dữ liệu đầu vào từ giá trị đã nhập
input_data = poly.transform([[price_ori_input, item_rating_input, favorite_input, total_sold_input]])

# Dự đoán price_actual
predicted_price_actual = model.predict(input_data)
print(f"Dự đoán price_actual: {predicted_price_actual[0]:.2f}")
