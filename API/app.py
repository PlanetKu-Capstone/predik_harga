from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Inisialisasi Flask app
app = Flask(__name__)

# Load model dan label encoder
model = joblib.load('C:/Users/bagus/msib/Capstone/predik_model/api/linear_model.pkl')  # Model XGBoost yang sudah dilatih
label_encoder = joblib.load('C:/Users/bagus/msib/Capstone/predik_model/api/label_encoder.pkl')  # LabelEncoder untuk fitur 'item'
features = joblib.load('C:/Users/bagus/msib/Capstone/predik_model/api/features.pkl')  # Fitur dari pelatihan

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data JSON dari request
        data = request.json

        # Validasi input data
        if 'item' not in data:
            return jsonify({'error': 'Field "item" is required.'}), 400

        # Encode fitur 'item'
        item = data['item']
        try:
            item_encoded = label_encoder.transform([item])[0]
        except ValueError:
            return jsonify({'error': f'Item "{item}" not recognized in the dataset.'}), 400

        # Buat DataFrame input dengan one-hot encoding
        input_data = pd.DataFrame(0, index=[0], columns=features)  # Template DataFrame
        input_data[f'item_{item}'] = 1  # One-hot encode kolom item

        # Prediksi harga
        predicted_price = model.predict(input_data)[0]

        # Kembalikan hasil prediksi
        return jsonify({
            'item': item,
            'predicted_price': round(predicted_price, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
