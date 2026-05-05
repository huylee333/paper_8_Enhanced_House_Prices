import pickle
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore") # Ẩn các cảnh báo của sklearn

# 1. Tải mô hình đã lưu
with open('bengaluru_house_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)

# 2. Tải cấu trúc cột để xử lý One-Hot Encoding cho Location
with open('columns.json', 'r') as f:
    data_columns = json.load(f)['data_columns']

# 3. Hàm dự đoán giá nhà
def predict_price(location, sqft, bath, bhk):
    # Khởi tạo một mảng toàn số 0 với kích thước bằng số lượng cột của mô hình
    x = np.zeros(len(data_columns))
    
    # Ba biến đầu tiên luôn là: Area, Bathroom, Bhk (dựa theo thứ tự tập X lúc train)
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    
    # Xử lý One-Hot Encoding cho địa điểm (Location)
    try:
        # Tìm vị trí index của location được nhập vào (chuyển về chữ thường để khớp với file json)
        loc_index = data_columns.index(location.lower())
        # Đánh dấu 1 tại vị trí index tương ứng
        x[loc_index] = 1
    except ValueError:
        # Nếu địa điểm không có trong danh sách (hoặc thuộc nhóm "other"), mảng dummies giữ nguyên giá trị 0
        pass 
        
    # Trả về giá trị dự đoán (Price)
    return round(model.predict([x])[0], 2)

# --- BẮT ĐẦU TEST MÔ HÌNH ---
print("Đang chạy test mô hình...")
print("=" * 50)

# Test 1: Căn hộ 1000 sqft, 2 phòng tắm, 2 phòng ngủ ở '1st Phase JP Nagar'
price_1 = predict_price('1st Phase JP Nagar', 1000, 2, 2)
print(f"Giá dự đoán Test 1: {price_1} Lakh (Trăm nghìn Rupee)")

# Test 2: Căn hộ 1000 sqft, 3 phòng tắm, 3 phòng ngủ ở '1st Phase JP Nagar' (Giá trị phải cao hơn test 1)
price_2 = predict_price('1st Phase JP Nagar', 1000, 3, 3)
print(f"Giá dự đoán Test 2: {price_2} Lakh")

# Test 3: Căn hộ 1000 sqft, 2 phòng tắm, 2 phòng ngủ ở 'Indira Nagar' (Khu vực khác)
price_3 = predict_price('Indira Nagar', 1000, 2, 2)
print(f"Giá dự đoán Test 3: {price_3} Lakh")

# --- ĐÁNH GIÁ ĐỘ CHÍNH XÁC MÔ HÌNH TRÊN TẬP TEST ---
print("\n" + "=" * 50)
print("ĐÁNH GIÁ ĐỘ CHÍNH XÁC MÔ HÌNH")
print("=" * 50)

# Tải lại dữ liệu và chuẩn bị tập test (giống Train.py)
df = pd.read_csv('cleaned_data_for_model.csv')
cols_to_drop = [col for col in ['area_type', 'availability', 'society', 'balcony'] if col in df.columns]
if cols_to_drop:
    df.drop(cols_to_drop, axis='columns', inplace=True)

dummies = pd.get_dummies(df.Location)
df_dummies = pd.concat([df, dummies.drop('other', axis='columns')], axis='columns')
df_model = df_dummies.drop('Location', axis='columns')

X = df_model.drop('Price', axis='columns')
y = df_model.Price

# Chia dữ liệu với cùng random_state để có cùng tập test như lúc train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Dự đoán trên tập test
y_pred = model.predict(X_test)

# Tính các chỉ số đánh giá
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Số lượng mẫu Train: {len(X_train)}")
print(f"Số lượng mẫu Test:  {len(X_test)}")
print(f"")
print(f"  R² Score (Hệ số xác định):       {r2:.4f}  ({r2*100:.2f}%)")
print(f"  MAE (Sai số tuyệt đối trung bình): {mae:.2f} Lakh")
print(f"  MSE (Sai số bình phương trung bình): {mse:.2f}")
print(f"  RMSE (Căn bậc hai MSE):            {rmse:.2f} Lakh")
print(f"")

# Đánh giá tổng quát
if r2 >= 0.85:
    print(f"✅ Mô hình đạt độ chính xác TỐT (R² = {r2:.4f} >= 0.85)")
elif r2 >= 0.70:
    print(f"⚠️ Mô hình đạt độ chính xác KHÁ (R² = {r2:.4f})")
else:
    print(f"❌ Mô hình cần cải thiện (R² = {r2:.4f} < 0.70)")