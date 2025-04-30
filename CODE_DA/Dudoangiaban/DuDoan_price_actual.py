import pandas as pd
import numpy as np

# Đọc dữ liệu
data = pd.read_csv("cleaned_shopee_data1.csv")

# Define the number of bins
num_bins = 5

price_ori_data = data['price_ori']
total_sold_data = data['total_sold']
# Cột item_rating
item_rating_data = data['item_rating']
# Cột favorite
favorite_data = data['favorite']

# Generate bin edges and labels cho cột price_ori
bin_edges = np.linspace(price_ori_data.min(), price_ori_data.max(), num_bins + 1)
labels = [
    f"<= {round(bin_edges[1], 2)}" if i == 0 else
    f"({round(bin_edges[i], 2)} - {round(bin_edges[i + 1], 2)}]" 
    for i in range(num_bins)
]

# Discretize the price_ori column
data['price_ori_binned'] = pd.cut(
    price_ori_data, 
    bins=bin_edges, 
    labels=labels, 
    include_lowest=True, 
    right=True
)

# Tạo khoảng (bins) và nhãn

bin_edges_total_sold = np.linspace(total_sold_data.min(), total_sold_data.max(), num_bins + 1)
labels_total_sold = [
    f"<= {round(bin_edges_total_sold[1], 2)}" if i == 0 else
    f"({round(bin_edges_total_sold[i], 2)} - {round(bin_edges_total_sold[i + 1], 2)}]"
    for i in range(num_bins)
]

# Rời rạc hóa dữ liệu
data['total_sold_binned'] = pd.cut(
    total_sold_data, 
    bins=bin_edges_total_sold, 
    labels=labels_total_sold, 
    include_lowest=True, 
    right=True
)


# Tạo khoảng (bins) và nhãn
num_bins_item_rating = 50

bin_edges = np.linspace(item_rating_data.min(), item_rating_data.max(), num_bins_item_rating + 1)
labels = [
    f"<= {round(bin_edges[1], 2)}" if i == 0 else
    f"({round(bin_edges[i], 2)} - {round(bin_edges[i + 1], 2)}]"
    for i in range(num_bins_item_rating)
]

# Rời rạc hóa cột item_rating
data['item_rating_binned'] = pd.cut(
    item_rating_data, 
    bins=bin_edges, 
    labels=labels, 
    include_lowest=True, 
    right=True
)

# Tạo khoảng (bins) và nhãn
bin_edges = np.linspace(favorite_data.min(), favorite_data.max(), num_bins + 1)
labels = [
    f"<= {round(bin_edges[1], 2)}" if i == 0 else
    f"({round(bin_edges[i], 2)} - {round(bin_edges[i + 1], 2)}]"
    for i in range(num_bins)
]
# Rời rạc hóa cột favorite
data['favorite_binned'] = pd.cut(
    favorite_data, 
    bins=bin_edges, 
    labels=labels, 
    include_lowest=True, 
    right=True
)

# Hiển thị kết quả
# Display the columns 'price_ori' and 'price_ori_binned'
#print(data[['price_ori', 'price_ori_binned']].head(20))
#Tất cả các giá trị trong mẫu này đều rơi vào nhóm <= 179.88, vì đây là nhóm đầu tiên và chiếm phần lớn dữ liệu
# Display the columns 'price_ori' and 'price_ori_binned'
#print(data[['total_sold', 'total_sold_binned']].head(20))
#print(data[['item_rating', 'item_rating_binned']].head(20))
#print(data[['favorite', 'favorite_binned']].head(20))

# Danh sách các cột cần hiển thị: các cột gốc và cột đã rời rạc hóa
discretized_columns = [ 'price_ori_binned', 'total_sold_binned', 'item_rating_binned', 'favorite_binned']

# Lấy dữ liệu từ các cột này và gom lại thành bảng
discretized_data = data[discretized_columns]

# Hiển thị 20 dòng đầu tiên của bảng dữ liệu đã gom
print(discretized_data.head(20))