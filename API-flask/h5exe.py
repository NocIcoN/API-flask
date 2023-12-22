from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)

# Memuat model (sesuaikan path sesuai lokasi model Anda)
model = tf.keras.models.load_model('model\Emotion Recognition From English text.h5')

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil data dari request
    data = request.json
    input_data = preprocess_data(data)

    # Lakukan prediksi
    prediction = model.predict(input_data)

    # Konversi hasil prediksi ke format yang diinginkan
    output_data = postprocess_prediction(prediction)

    return jsonify(output_data)

def preprocess_data(data):
    # Fungsi untuk memproses data input agar sesuai dengan format model
    # Implementasi fungsi ini tergantung pada kebutuhan model Anda
    pass

def postprocess_prediction(prediction):
    # Fungsi untuk mengolah hasil prediksi sebelum dikirimkan kembali
    # Implementasi fungsi ini tergantung pada kebutuhan Anda
    pass

if __name__ == '__main__':
    app.run(debug=True)
