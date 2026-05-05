import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import json

# Tải dữ liệu sạch
df = pd.read_csv('cleaned_data_for_model.csv')

# Loại bỏ các cột không cần thiết cho việc huấn luyện (còn sót từ dataset gốc)
cols_to_drop = [col for col in ['area_type', 'availability', 'society', 'balcony'] if col in df.columns]
if cols_to_drop:
    df.drop(cols_to_drop, axis='columns', inplace=True)
    print(f"Đã loại bỏ các cột không cần thiết: {cols_to_drop}")

# 1. Chuyển đổi dữ liệu phân loại (Location) thành số (One-Hot Encoding)
# Linear Regression không thể hiểu dạng chữ, nên ta phải tạo biến giả (dummy variables)
dummies = pd.get_dummies(df.Location)
df_dummies = pd.concat([df, dummies.drop('other', axis='columns')], axis='columns')
df_model = df_dummies.drop('Location', axis='columns')

# 2. Định nghĩa biến độc lập (X) và biến phụ thuộc (y)
X = df_model.drop('Price', axis='columns')
y = df_model.Price

# 3. Chia dữ liệu thành tập Huấn luyện (80%) và tập Kiểm tra (20%)[cite: 1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

print(f"Số lượng mẫu Train: {len(X_train)}")
print(f"Số lượng mẫu Test:  {len(X_test)}")
print("=" * 85)

# ============================================================
# 4. Khởi tạo và Huấn luyện tất cả các mô hình
# ============================================================

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression":  Ridge(alpha=1.0),
    "Lasso Regression":  Lasso(alpha=0.1, max_iter=10000),
    "Decision Tree":     DecisionTreeRegressor(random_state=10),
    "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=10),
}

results = []

for name, model in models.items():
    print(f"\n>> Đang huấn luyện: {name}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    results.append({
        "name": name,
        "model": model,
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
    })
    
    print(f"   R² = {r2:.4f} | MAE = {mae:.2f} | RMSE = {rmse:.2f}")

# ============================================================
# 5. Bảng so sánh kết quả tất cả các mô hình
# ============================================================
print("\n" + "=" * 85)
print("BẢNG SO SÁNH KẾT QUẢ CÁC MÔ HÌNH")
print("=" * 85)
print(f"{'Mô hình':<25} {'R² Score':>10} {'MAE (Lakh)':>12} {'RMSE (Lakh)':>13}")
print("-" * 85)

for r in results:
    marker = " ★" if r["r2"] == max(x["r2"] for x in results) else ""
    print(f"{r['name']:<25} {r['r2']:>10.4f} {r['mae']:>12.2f} {r['rmse']:>13.2f}{marker}")

print("-" * 85)

# Tìm mô hình tốt nhất (R² cao nhất)
best = max(results, key=lambda x: x["r2"])
print(f"\n🏆 Mô hình tốt nhất: {best['name']} (R² = {best['r2']:.4f})")

# ============================================================
# 6. Lưu mô hình tốt nhất để triển khai lên Web Server (Flask)[cite: 1]
# ============================================================
with open('bengaluru_house_prices_model.pickle', 'wb') as f:
    pickle.dump(best["model"], f)

# Lưu lại danh sách các cột (locations) để Web Server biết thứ tự truyền vào dự đoán
columns = {
    'data_columns': [col.lower() for col in X.columns]
}
with open("columns.json", "w") as f:
    f.write(json.dumps(columns))

print(f"\n✅ Đã lưu mô hình '{best['name']}' thành công. Sẵn sàng cho triển khai!")