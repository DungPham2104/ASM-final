
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Đọc dữ liệu
df = pd.read_csv("amazon.csv")

# Tính toán hệ số tương quan
plt.figure(figsize=(8, 6))
sns.heatmap(df[['discounted_price', 'discount_percentage', 'rating_count', 'rating']].corr(),
            annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Ma trận tương quan')
plt.show()

# Biểu đồ phân phối giá sau giảm
plt.figure(figsize=(10, 6))
sns.histplot(df['discounted_price'], bins=30, kde=True, color='#1f77b4')
plt.title('Phân phối giá sau giảm')
plt.xlabel('Giá (INR)')
plt.ylabel('Số lượng')
plt.grid(True, alpha=0.3)
plt.show()

# Điểm đánh giá trung bình theo danh mục
plt.figure(figsize=(12, 6))
avg_rating = df.groupby('category_level_1')['rating'].mean().sort_values()
sns.barplot(x=avg_rating.index, y=avg_rating.values, color='#ff7f0e')
plt.title('Điểm đánh giá trung bình theo danh mục')
plt.xlabel('Danh mục')
plt.ylabel('Điểm đánh giá trung bình')
plt.xticks(rotation=45, ha='right')
plt.grid(True, axis='y', alpha=0.3)
plt.show()

# Mối quan hệ giữa mức giảm giá và số lượng đánh giá
plt.figure(figsize=(10, 6))
sns.scatterplot(x='discount_percentage', y='rating_count', data=df, color='#2ca02c', alpha=0.6)
plt.title('Mức giảm giá vs Số lượng đánh giá')
plt.xlabel('Mức giảm giá (%)')
plt.ylabel('Số lượng đánh giá')
plt.grid(True, alpha=0.3)
plt.show()

# Top 5 sản phẩm có số lượng đánh giá cao nhất
top_products = df.nlargest(5, 'rating_count')[['product_name', 'rating_count']]
plt.figure(figsize=(12, 6))
sns.barplot(x='rating_count', y='product_name', data=top_products, color='#d62728')
plt.title('Top 5 sản phẩm có số lượng đánh giá cao nhất')
plt.xlabel('Số lượng đánh giá')
plt.ylabel('Tên sản phẩm')
plt.grid(True, axis='x', alpha=0.3)
plt.show()

# Giá sau giảm vs Điểm đánh giá
plt.figure(figsize=(10, 6))
sns.scatterplot(x='discounted_price', y='rating', data=df, color='#9467bd', alpha=0.6)
plt.title('Giá sau giảm vs Điểm đánh giá')
plt.xlabel('Giá sau giảm (INR)')
plt.ylabel('Điểm đánh giá')
plt.grid(True, alpha=0.3)
plt.show()

# Mô hình hồi quy tuyến tính để dự đoán rating
df_model = df.dropna(subset=['discounted_price', 'discount_percentage', 'rating_count', 'rating'])
X = df_model[['discounted_price', 'discount_percentage', 'rating_count']]
y = df_model['rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Hệ số đặc trưng
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("\nHệ số đặc trưng:")
print(coefficients)

# So sánh giá trị thực tế và dự đoán
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='#1f77b4', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Giá trị thực tế vs Dự đoán (Rating)')
plt.xlabel('Rating thực tế')
plt.ylabel('Rating dự đoán')
plt.grid(True, alpha=0.3)
plt.show()

# Phân phối lỗi dự đoán
errors = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(errors, bins=30, kde=True, color='#ff7f0e')
plt.title('Phân phối lỗi dự đoán')
plt.xlabel('Lỗi (Rating thực tế - Dự đoán)')
plt.ylabel('Số lượng')
plt.grid(True, alpha=0.3)
plt.show()
