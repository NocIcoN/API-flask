from flask import Flask, request, jsonify
import tensorflow as tf
import firebase_admin
from firebase_admin import credentials, firestore

# Inisialisasi Flask App
app = Flask(__name__)

# Memuat model ML (Gantikan dengan lokasi file model Anda)
model = tf.keras.models.load_model('model\Emotion Recognition From English text.h5')

# Inisialisasi Firestore
cred = credentials.Certificate("JSON\quick-formula-405704-firebase-adminsdk-cnzjy-80571da287.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

@app.route('/process', methods=['POST'])
def process_data():
    # Ambil data input dari Firestore
    input_data = db.collection('inputs').document('input_doc').get().to_dict()

    # Proses data dengan model ML
    processed_data = model.predict(input_data)

    # Kirim hasil kembali ke Firestore
    db.collection('outputs').document('output_doc').set(processed_data)

    return jsonify({"status": "Success", "data": processed_data}), 200

if __name__ == '__main__':
    app.run(debug=True)
