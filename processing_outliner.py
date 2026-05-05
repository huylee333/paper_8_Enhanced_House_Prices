import pandas as pd
import numpy as np

# Tải lại dữ liệu từ bước 1
df6 = pd.read_csv('processed_data_step1.csv')

# 1. Loại bỏ ngoại lai theo tỷ lệ Diện tích / Số phòng ngủ (Area / Bhk)
# Ngưỡng tối thiểu là 300 sqft cho mỗi phòng ngủ[cite: 1]
df7 = df6[~(df6.Area/df6.Bhk < 300)]
print(f"Số lượng dữ liệu sau khi xóa ngoại lai Area/Bhk: {df7.shape}")

# 2. Loại bỏ ngoại lai Giá trên mỗi mét vuông (Price_per_sqft) bằng Mean và Std
# Áp dụng cho từng vị trí cụ thể[cite: 1]
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('Location'):
        m = np.mean(subdf.Price_per_sqft)
        st = np.std(subdf.Price_per_sqft)
        reduced_df = subdf[(subdf.Price_per_sqft > (m-st)) & (subdf.Price_per_sqft <= (m+st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

df8 = remove_pps_outliers(df7)
print(f"Số lượng dữ liệu sau khi xóa ngoại lai Price_per_sqft: {df8.shape}")

# (Tùy chọn) Hàm remove_bhk_outliers được nhắc đến trong bài báo để bỏ các căn 3 ngủ rẻ hơn 2 ngủ cùng diện tích[cite: 1].
# Để giữ code ngắn gọn, tôi tập trung vào 3 bộ lọc chính được trích dẫn rõ mã nguồn nhất.

# 3. Loại bỏ ngoại lai theo Số phòng tắm (Bathroom)
# Số phòng tắm không được lớn hơn tổng số phòng ngủ + 2[cite: 1]
df10 = df8[df8.Bathroom < df8.Bhk + 2]
print(f"Số lượng dữ liệu sau khi xóa ngoại lai Bathroom: {df10.shape}")

# Lưu dữ liệu sạch để đưa vào mô hình
df10.drop(['Room', 'Price_per_sqft'], axis='columns', inplace=True) # Bỏ các cột không dùng để train
df10.to_csv('cleaned_data_for_model.csv', index=False)
print("Đã xong Notebook 2")