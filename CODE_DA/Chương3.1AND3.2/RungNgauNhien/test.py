
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


# Đọc dữ liệu từ file CSV (giả sử dữ liệu của bạn được lưu trong file data.csv)
data = pd.read_csv('data.csv')

# Chọn các đặc trưng và biến mục tiêu (ví dụ: dự đoán số lượng bán được dựa trên giá và danh mục)
X = data[['price', 'item_cate']]
y = data['total_sold']

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tạo mô hình Rừng ngẫu nhiên
rf = RandomForestRegressor(n_estimators=100)

y_pred = model.predict(X_test)

cm = plot_confusion_matrix(y_true, y_pred)

# Huấn luyện mô hình
rf.fit(X_train, y_train)


# Dự đoán
y_pred = rf.predict(X_test)
plot_confusion_matrix(cm, classes=['Not Popular', 'Popular'])

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()