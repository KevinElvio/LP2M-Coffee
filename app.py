# from flask import Flask, render_template, request, jsonify
# import pickle
# import joblib
# import numpy as np
# import pandas as pd

# app = Flask(__name__)

# # Muat model dan scaler
# # Pastikan path ke file ini benar
# try:
#     model = pickle.load(open('model.pkl', 'rb'))
#     scaler = joblib.load('scaler_x.joblib')
#     # Definisikan le di sini atau muat juga jika sudah disimpan
#     # Contoh sederhana jika Anda tidak menyimpan LabelEncoder
#     # Pastikan urutan ini sesuai dengan LabelEncoder Anda saat training
#     labels = ["Arabica", "Robusta"]
#     print("Model and scaler loaded successfully.")
# except Exception as e:
#     print(f"Error loading model or scaler: {e}")
#     model = None
#     scaler = None
#     labels = ["Unknown"] # Default labels if loading fails


# @app.route('/')
# def index():
#     # return render_template('index.html')
#     return ('Hello World!')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if model is None or scaler is None:
#         return jsonify({'error': 'Model or scaler not loaded on the server. Please check server logs.'}), 500

#     data = request.get_json(force=True)
#     print("Received data:", data)

#     features = ['Aroma','Flavor','Aftertaste','Acidity', 'Sweetness']# Sesuaikan dengan urutan fitur Anda

#     # Buat numpy array dari data input dengan validasi sederhana
#     input_values = []
#     try:
#         for feature in features:
#             value = data.get(feature)
#             if value is None:
#                  return jsonify({'error': f"Missing feature: {feature}"}), 400 # Bad Request
#             try:
#                 # Coba konversi ke float
#                 input_values.append(float(value))
#             except ValueError:
#                  return jsonify({'error': f"Invalid value for feature {feature}: {value}. Expected a number."}), 400

#     except Exception as e:
#          return jsonify({'error': f"Error processing input data: {e}"}), 400


#     input_data = np.array([input_values])

#     # Lakukan scaling pada data input
#     try:
#         scaled_data = scaler.transform(input_data)
#     except Exception as e:
#         print(f"Error during scaling: {e}")
#         return jsonify({'error': f"Error processing data for prediction: {e}"}), 500


#     # Reshape untuk LSTM (sesuai dengan cara Anda melatih model)
#     # Periksa bentuk data sebelum reshape jika perlu debugging
#     # print("Shape before reshape:", scaled_data.shape)
#     scaled_data_lstm = np.expand_dims(scaled_data, axis=1)
#     # print("Shape after reshape:", scaled_data_lstm.shape)


#     # Lakukan prediksi
#     try:
#         prediction_proba = model.predict(scaled_data_lstm)
#         predicted_class_index = np.argmax(prediction_proba, axis=1)[0]

#         # Menggunakan labels yang sudah dimuat
#         predicted_label = labels[predicted_class_index]

#     except IndexError:
#          predicted_label = "Unknown Species (Index out of bounds)"
#          print(f"Warning: Predicted class index {predicted_class_index} out of bounds for labels list of size {len(labels)}")
#     except Exception as e:
#         print(f"Error during prediction: {e}")
#         return jsonify({'error': f"Error during model prediction: {e}"}), 500


#     return jsonify({'prediction': predicted_label})

# if __name__ == '__main__':
#     # Untuk deployment, ubah debug=True menjadi False
#     # dan gunakan server produksi seperti Gunicorn atau uWSGI
#     app.run(debug=True)

