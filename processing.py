# Import các thư viện cần thiết
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Tải dữ liệu
# Giả sử bạn đã tải bộ dữ liệu Bengaluru House Price Data từ Kaggle và lưu tên là 'bengaluru_house_prices.csv'
df1 = pd.read_csv('Bengaluru_House_Data.csv')

# Theo bài báo, các cột quan tâm gồm: Location, Room (Size), Area (total_sqft), Bathroom, Price
# Đổi tên cột cho giống với paper để dễ theo dõi
df2 = df1.rename(columns={'total_sqft': 'Area', 'size': 'Room', 'bath': 'Bathroom', 'price': 'Price', 'location': 'Location'})

# 2. Xử lý giá trị bị thiếu (Missing Values)
# Sử dụng giá trị trung vị (median) để điền vào các giá trị bị thiếu của cột Bathroom[cite: 1]
median_bath = df2['Bathroom'].median()
df2['Bathroom'].fillna(median_bath, inplace=True)

# Bỏ qua các giá trị null còn lại cho đơn giản
df3 = df2.dropna()

# 3. Tạo đặc trưng mới (Feature Engineering)
# Tạo cột 'Bhk' từ cột 'Room'[cite: 1]
df5 = df3.copy()
df5["Bhk"] = df3["Room"].apply(lambda x: int(x.split(" ")[0]) if isinstance(x, str) and " " in x else None)

# Chuyển đổi cột Area (total_sqft) sang dạng float (bỏ qua các giá trị dạng range như '1133 - 1384')
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
df5 = df5[df5['Area'].apply(is_float)]
df5['Area'] = df5['Area'].astype(float)

# 4. Giảm chiều dữ liệu (Dimensionality Reduction) cho Location
# Gộp các vị trí có ít hơn 10 dữ liệu thành nhóm "other"[cite: 1]
df5.Location = df5.Location.apply(lambda x: x.strip() if isinstance(x, str) else x)
location_stats = df5.groupby('Location')['Location'].agg('count').sort_values(ascending=False)
location_stats_less_than_10 = location_stats[location_stats <= 10]

df6 = df5.copy()
df6['Location'] = df6['Location'].apply(lambda x: "other" if x in location_stats_less_than_10 else x)

# 5. Vẽ biểu đồ tương quan (Correlation Heatmap)
# Tạo cột Price_per_sqft để dùng cho bước sau (Giá trị Price trong dataset thường tính bằng Lakh - 100,000 Rupee)
df6['Price_per_sqft'] = df6['Price'] * 100000 / df6['Area']

plt.figure(figsize=(8,6))
sns.heatmap(df6[['Area', 'Bathroom', 'Price', 'Bhk', 'Price_per_sqft']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Lưu dữ liệu đã xử lý để dùng cho Notebook 2
df6.to_csv('processed_data_step1.csv', index=False)
print("Đã xong Notebook 1")