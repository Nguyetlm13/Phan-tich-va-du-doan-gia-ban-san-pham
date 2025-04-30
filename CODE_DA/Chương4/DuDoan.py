import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# Bước 1: Đọc dữ liệu
data = pd.read_csv('cleaned_shopee_data.csv')  # Thay bằng đường dẫn file của bạn
columns_of_interest = ['price_ori', 'total_sold', 'favorite', 'item_rating', 'price_actual']
data = data[columns_of_interest]

# Bước 2: Chia dữ liệu
X = data[['price_ori', 'total_sold', 'favorite', 'item_rating']]
y = data['price_actual']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bước 3: Tạo mô hình hồi quy đa thức
degree = 2
model = make_pipeline(PolynomialFeatures(degree), StandardScaler(), LinearRegression())
model.fit(X_train, y_train)

# Bước 4: Đánh giá mô hình
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Bước 5: Hàm dự đoán
def predict_price(price_ori, total_sold, favorite, item_rating):
    # Tạo DataFrame đầu vào với tên cột giống như dữ liệu huấn luyện
    input_data = pd.DataFrame([[price_ori, total_sold, favorite, item_rating]], 
                              columns=['price_ori', 'total_sold', 'favorite', 'item_rating'])
     # Đặt ngưỡng giá tối thiểu (ví dụ: 10% giá gốc hoặc một giá trị cố định)
    min_price = 0.1 * price_ori  # Giá bán tối thiểu là 10% giá gốc
    
    # Tạo các đặc trưng bậc 2 (polynomial features)
    input_poly = model.named_steps['polynomialfeatures'].transform(input_data)
    
    # Chuẩn hóa các đặc trưng
    input_poly_scaled = model.named_steps['standardscaler'].transform(input_poly)
    
    # Dự đoán giá
    prediction = model.named_steps['linearregression'].predict(input_poly_scaled)
        # Đảm bảo giá trị dự đoán không âm
    return max(prediction[0], 0)

    return prediction[0]

# Bước 6: Nhập giá trị từ người dùng và dự đoán
price_ori = float(input("Nhập giá gốc sản phẩm (price_ori): "))
total_sold = float(input("Nhập tổng số lượng bán ra (total_sold): "))
favorite = int(input("Nhập số lượng yêu thích (favorite): "))
item_rating = float(input("Nhập đánh giá sản phẩm (item_rating): "))

predicted_price = predict_price(price_ori, total_sold, favorite, item_rating)
print(f"Dự đoán giá thực tế (price_actual): {predicted_price:.2f}")
