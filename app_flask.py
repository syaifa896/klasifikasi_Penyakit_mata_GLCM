# app.py

from flask import Flask, request, render_template, send_from_directory
import cv2
import numpy as np
import joblib
import json
import os
from skimage.feature import graycomatrix, graycoprops
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Pastikan folder 'uploads' ada
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# --- Muat Model, Scaler, dan Nama Kelas saat aplikasi dimulai ---
# Variabel global untuk model, scaler, dan nama kelas
model = None
scaler = None
class_names = []

try:
    model = joblib.load('klasifikasi_mata_model.pkl')
    scaler = joblib.load('scaler_for_glcm.pkl') # Muat scaler
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    print("Model, Scaler, dan Nama kelas berhasil dimuat.")
except FileNotFoundError:
    print("Error: Salah satu file (model/scaler/nama kelas) tidak ditemukan.")
    print("Pastikan Anda sudah menjalankan 'train_model.py' terlebih dahulu.")
except Exception as e:
    print(f"Terjadi kesalahan saat memuat model/scaler/nama kelas: {e}")

# --- Fungsi-fungsi pengolahan citra (harus sama persis dengan train_model.py) ---
# PASTIKAN FUNGSI INI BERADA DI LUAR FUNGSI upload_file DAN MENERIMA image_path
def preprocess_image(image_path, target_size=(256, 256), method='clahe'):
    """
    Melakukan pre-processing pada citra dari sebuah path file.
    Args:
        image_path (str): Path lengkap menuju file citra.
        target_size (tuple): Ukuran target (lebar, tinggi) untuk resize citra.
        method (str): Metode pre-processing tambahan ('gaussian_blur', 'median_filter', 'clahe', atau 'none').
    Returns:
        numpy.ndarray: Citra yang sudah diproses (grayscale), atau None jika gagal membaca citra.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Tidak dapat membaca citra dari {image_path}") # Debugging
        return None

    img = cv2.resize(img, target_size)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    processed_img = gray_img
    if method == 'gaussian_blur':
        processed_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    elif method == 'median_filter':
        processed_img = cv2.medianBlur(gray_img, 5)
    elif method == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        processed_img = clahe.apply(gray_img)
    elif method == 'none':
        processed_img = gray_img

    return processed_img

def extract_glcm_features(image_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256):
    """
    Mengekstraksi fitur GLCM dari citra grayscale.
    """
    image_gray = image_gray.astype(int)
    glcm = graycomatrix(image_gray,
                        distances=distances,
                        angles=angles,
                        levels=levels,
                        symmetric=True,
                        normed=True)
    features = []
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    for prop in properties:
        features.extend(graycoprops(glcm, prop).flatten())
    return np.array(features)
# --- Akhir fungsi pengolahan citra ---

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    prediction_result = None
    uploaded_filename = None
    processed_image_path = None
    message = None
    selected_method_from_ui = 'clahe' # Default method untuk tampilan awal

    if request.method == 'POST':
        selected_method_from_ui = request.form.get('preprocess_method', 'clahe')

        if 'file' not in request.files:
            message = 'Tidak ada bagian file di request.'
        file = request.files['file']
        if file.filename == '':
            message = 'Tidak ada file yang dipilih.'

        if not message and file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            uploaded_filename = filename

            if model is not None and scaler is not None:
                # Panggil fungsi preprocess_image yang sudah ada di luar scope ini
                # dan pastikan ia menerima filepath
                processed_img_array = preprocess_image(filepath, method=selected_method_from_ui)
                
                if processed_img_array is not None:
                    # Simpan gambar yang sudah diproses untuk ditampilkan di UI
                    # Pastikan nama file unik untuk gambar yang diproses
                    processed_filename_for_display = 'processed_' + os.path.splitext(filename)[0] + '.png' # Gunakan .png untuk konsistensi
                    processed_display_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename_for_display)
                    cv2.imwrite(processed_display_path, processed_img_array)
                    processed_image_path = processed_filename_for_display # Ini yang akan dikirim ke template

                    glcm_features = extract_glcm_features(processed_img_array)
                    
                    # Terapkan scaler yang sudah dilatih!
                    glcm_features_scaled = scaler.transform(glcm_features.reshape(1, -1))

                    prediction = model.predict(glcm_features_scaled)
                    prediction_result = class_names[prediction[0]]
                else:
                    message = 'Gagal memproses gambar. Pastikan file gambar valid.'
            else:
                message = 'Model atau scaler belum dimuat. Silakan periksa log server.'
        elif not message: # Jika file tidak diizinkan
            message = 'Jenis file tidak diizinkan.'
    
    return render_template('index.html',
                           prediction=prediction_result,
                           filename=uploaded_filename,
                           processed_image_path=processed_image_path, 
                           message=message,
                           selected_method=selected_method_from_ui)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)