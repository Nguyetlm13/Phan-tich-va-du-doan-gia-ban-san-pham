from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

# Bước 1: Đọc dữ liệu
data = pd.read_csv('cleaned_shopee_data.csv')  # Thay bằng đường dẫn file của bạn

# Selecting relevant columns for modeling
columns_of_interest = ['price_ori', 'total_sold', 'favorite', 'item_rating', 'price_actual']
data = data[columns_of_interest].dropna()

# Splitting the data
X = data[['price_ori', 'total_sold', 'favorite', 'item_rating']]
y = data['price_actual']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating polynomial features
degree = 2
poly = PolynomialFeatures(degree)
X_poly = poly.fit_transform(X_train)

# Standardizing features
scaler = StandardScaler()
X_poly_scaled = scaler.fit_transform(X_poly)

# Fitting the linear regression model
model = LinearRegression()
model.fit(X_poly_scaled, y_train)

# Extracting coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_

# Mapping coefficients to their respective feature combinations
feature_names = poly.get_feature_names_out(X.columns)
coeff_dict = dict(zip(feature_names, coefficients))

intercept, coeff_dict