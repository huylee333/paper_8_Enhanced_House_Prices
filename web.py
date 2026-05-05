from flask import Flask, request, jsonify, render_template_string
import pickle
import json
import numpy as np

app = Flask(__name__)

# Giao diện HTML cho trang chủ
HTML_PAGE = '''
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dự đoán giá nhà Bengaluru</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #fff;
        }
        .container {
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 20px;
            padding: 40px;
            width: 480px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        h1 {
            text-align: center;
            margin-bottom: 8px;
            font-size: 1.6em;
            background: linear-gradient(90deg, #a78bfa, #60a5fa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            text-align: center;
            color: rgba(255,255,255,0.5);
            margin-bottom: 28px;
            font-size: 0.9em;
        }
        label {
            display: block;
            margin-bottom: 6px;
            font-weight: 500;
            color: rgba(255,255,255,0.8);
            font-size: 0.9em;
        }
        input, select {
            width: 100%;
            padding: 12px 16px;
            margin-bottom: 18px;
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 10px;
            background: rgba(255,255,255,0.06);
            color: #fff;
            font-size: 1em;
            outline: none;
            transition: border-color 0.3s;
        }
        input:focus, select:focus {
            border-color: #a78bfa;
        }
        select option { background: #302b63; color: #fff; }
        button {
            width: 100%;
            padding: 14px;
            border: none;
            border-radius: 12px;
            background: linear-gradient(135deg, #a78bfa, #60a5fa);
            color: #fff;
            font-size: 1.05em;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(167, 139, 250, 0.4);
        }
        .result {
            margin-top: 24px;
            text-align: center;
            padding: 20px;
            border-radius: 12px;
            background: rgba(96, 165, 250, 0.12);
            border: 1px solid rgba(96, 165, 250, 0.25);
            display: none;
        }
        .result .price {
            font-size: 2em;
            font-weight: 700;
            background: linear-gradient(90deg, #34d399, #60a5fa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .result .unit {
            color: rgba(255,255,255,0.5);
            font-size: 0.85em;
            margin-top: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏠 Dự đoán giá nhà</h1>
        <p class="subtitle">Bengaluru House Price Prediction</p>
        <form id="priceForm">
            <label for="total_sqft">Diện tích (sqft)</label>
            <input type="number" id="total_sqft" name="total_sqft" placeholder="Ví dụ: 1000" required>

            <label for="bhk">Số phòng ngủ (BHK)</label>
            <select id="bhk" name="bhk">
                <option value="1">1</option>
                <option value="2" selected>2</option>
                <option value="3">3</option>
                <option value="4">4</option>
                <option value="5">5</option>
            </select>

            <label for="bath">Số phòng tắm</label>
            <select id="bath" name="bath">
                <option value="1">1</option>
                <option value="2" selected>2</option>
                <option value="3">3</option>
                <option value="4">4</option>
                <option value="5">5</option>
            </select>

            <label for="location">Vị trí</label>
            <select id="location" name="location">
                <option value="">Đang tải...</option>
            </select>

            <button type="submit">Dự đoán giá</button>
        </form>
        <div class="result" id="result">
            <div class="price" id="priceValue"></div>
            <div class="unit">Lakh (Trăm nghìn Rupee)</div>
        </div>
    </div>
    <script>
        // Tải danh sách locations khi mở trang
        fetch('/get_location_names')
            .then(r => r.json())
            .then(data => {
                const sel = document.getElementById('location');
                sel.innerHTML = '';
                data.locations.forEach(loc => {
                    const opt = document.createElement('option');
                    opt.value = loc;
                    opt.textContent = loc;
                    sel.appendChild(opt);
                });
            });

        // Submit form
        document.getElementById('priceForm').addEventListener('submit', function(e) {
            e.preventDefault();
            const formData = new FormData(this);
            fetch('/predict_home_price', { method: 'POST', body: formData })
                .then(r => r.json())
                .then(data => {
                    document.getElementById('priceValue').textContent = data.estimated_price;
                    document.getElementById('result').style.display = 'block';
                })
                .catch(err => alert('Lỗi: ' + err));
        });
    </script>
</body>
</html>
'''

# Khai báo biến toàn cục
__model = None
__data_columns = None
__locations = None

def load_saved_artifacts():
    print("Đang tải các file mô hình... bắt đầu")
    global __data_columns
    global __locations
    global __model

    # Tải file json chứa cấu trúc cột
    with open("./columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]  # Bỏ qua 3 cột đầu là area, bath, bhk

    # Tải file pickle chứa mô hình[cite: 1]
    if __model is None:
        with open('./bengaluru_house_prices_model.pickle', 'rb') as f:
            __model = pickle.load(f)
    print("Tải mô hình thành công!")

def get_estimated_price(location, sqft, bath, bhk):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)

# Trang chủ - Giao diện Web
@app.route('/')
def home():
    return render_template_string(HTML_PAGE)

# API 1: Lấy danh sách các địa điểm để hiển thị trên giao diện Web HTML[cite: 1]
@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    response = jsonify({
        'locations': __locations
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

# API 2: Nhận dữ liệu từ form HTML và trả về giá dự đoán[cite: 1]
@app.route('/predict_home_price', methods=['GET', 'POST'])
def predict_home_price():
    # Lấy thông số từ request (post form)
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])

    response = jsonify({
        'estimated_price': get_estimated_price(location, total_sqft, bath, bhk)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == "__main__":
    print("Bắt đầu khởi động Python Flask Server...")
    load_saved_artifacts()
    # Chạy server ở cổng 5000
    app.run(port=5000)