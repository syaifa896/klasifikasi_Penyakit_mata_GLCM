# train_model.py

# 1. Import Library yang Dibutuhkan
import cv2
import numpy as np
import os
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler # Tambahkan ini untuk Feature Scaling
import matplotlib.pyplot as plt
import joblib # Untuk menyimpan dan memuat model
import json # Untuk menyimpan nama-nama kelas

# 2. Fungsi untuk Pre-processing Citra (Materi Tambahan #1: Filtering/Enhancement)
def preprocess_image(image_path, target_size=(256, 256), method='clahe'):
    """
    Melakukan pre-processing pada citra.
    Args:
        image_path (str): Path lengkap menuju file citra.
        target_size (tuple): Ukuran target (lebar, tinggi) untuk resize citra.
        method (str): Metode pre-processing tambahan ('gaussian_blur', 'median_filter', 'clahe', atau 'none').
    Returns:
        numpy.ndarray: Citra yang sudah diproses (grayscale), atau None jika gagal membaca citra.
    """
    img = cv2.imread(image_path)
    if img is None:
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
    elif method == 'none': # Opsi jika hanya ingin grayscale
        processed_img = gray_img

    return processed_img

# 3. Fungsi untuk Ekstraksi Fitur GLCM
def extract_glcm_features(image_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256):
    """
    Mengekstraksi fitur GLCM dari citra grayscale.
    Args:
        image_gray (numpy.ndarray): Citra grayscale.
        distances (list): Jarak antar piksel untuk GLCM.
        angles (list): Sudut untuk GLCM dalam radian.
        levels (int): Jumlah gray level dalam citra.
    Returns:
        numpy.ndarray: Array 1D dari fitur-fitur GLCM yang dihitung.
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

# 4. Fungsi Utama untuk Memuat Dataset dan Ekstraksi Fitur
def load_dataset_and_extract_features(dataset_path, pre_process_method='clahe'):
    """
    Memuat dataset citra dari folder, melakukan pre-processing, dan mengekstraksi fitur GLCM.
    Returns:
        tuple: (fitur_data, label_data, class_names)
    """
    all_features = []
    all_labels = []
    class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    print(f"Ditemukan kelas: {class_names}")

    for label_idx, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_path, class_name)
        print(f"\nMemproses citra dari kelas: '{class_name}' ({label_idx})")
        image_count = 0
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                processed_img = preprocess_image(image_path, method=pre_process_method)
                if processed_img is not None:
                    glcm_features = extract_glcm_features(processed_img)
                    all_features.append(glcm_features)
                    all_labels.append(label_idx)
                    image_count += 1
        print(f"  -> {image_count} citra diproses untuk kelas '{class_name}'.")

    return np.array(all_features), np.array(all_labels), class_names

# 5. Eksekusi Utama Program Pelatihan
if __name__ == "__main__":
   
    dataset_folder = 'D:\SYAIFATURROHMAN\SMT 6\TUGAS\PENGOLAHAN CITRA\PRAKTIKUM\PC app\dataset2'

    # Pilih metode pre-processing untuk training (harus sama dengan yang digunakan di app.py)
    selected_pre_process_method = 'clahe'

    print("--- Memulai Proses Pengumpulan Data dan Ekstraksi Fitur ---")
    X, y, class_names = load_dataset_and_extract_features(dataset_folder, pre_process_method=selected_pre_process_method)

    if X.size == 0:
        print("\n[ERROR] Tidak ada fitur yang diekstraksi. Pastikan path dataset benar.")
    else:
        print(f"\n--- Data Berhasil Dimuat ---")
        print(f"Total citra diproses: {len(X)}")
        print(f"Dimensi dataset fitur (jumlah_sampel, jumlah_fitur): {X.shape}")
        print(f"Dimensi label: {y.shape}")
        print(f"Nama Kelas (sesuai indeks): {class_names}")

        # 6. Bagi Data menjadi Training dan Testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print("\n--- Membagi Data Training dan Testing ---")
        print(f"Jumlah data training: {len(X_train)} sampel")
        print(f"Jumlah data testing: {len(X_test)} sampel")

        # 7. Feature Scaling (PENTING untuk SVM dan meningkatkan akurasi)
        print("\n--- Melakukan Feature Scaling ---")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print("Feature scaling selesai.")

        # 8. Latih Model Klasifikasi (SVC)
        print("\n--- Melatih Model Klasifikasi ---")
        model = SVC(kernel='linear', random_state=42) # Bisa coba 'rbf' atau classifier lain
        model.fit(X_train_scaled, y_train)
        print("Model klasifikasi berhasil dilatih.")

        # 9. Evaluasi Model
        print("\n--- Melakukan Evaluasi Model ---")
        y_pred = model.predict(X_test_scaled)

        print("\nLaporan Klasifikasi:")
        print(classification_report(y_test, y_pred, target_names=class_names))

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Akurasi Model Keseluruhan: {accuracy * 100:.2f}%")

        # 10. Menyimpan Model dan Scaler serta Nama Kelas
        model_filename = 'klasifikasi_mata_model.pkl'
        scaler_filename = 'scaler_for_glcm.pkl' # Simpan scaler juga!
        class_names_filename = 'class_names.json'

        joblib.dump(model, model_filename)
        joblib.dump(scaler, scaler_filename) # Simpan scaler
        with open(class_names_filename, 'w') as f:
            json.dump(class_names, f)

        print(f"\nModel berhasil disimpan sebagai '{model_filename}'")
        print(f"Scaler berhasil disimpan sebagai '{scaler_filename}'")
        print(f"Nama kelas berhasil disimpan sebagai '{class_names_filename}'")

        # 11. (Opsional) Visualisasi Contoh Citra dan Hasil Pre-processing
        print("\n--- Visualisasi Contoh Citra dan Hasil Pre-processing ---")
        if len(class_names) > 0:
            sample_class_path = os.path.join(dataset_folder, class_names[0])
            if os.path.exists(sample_class_path) and len(os.listdir(sample_class_path)) > 0:
                sample_image_name = os.listdir(sample_class_path)[0]
                sample_image_path = os.path.join(sample_class_path, sample_image_name)

                original_img_color = cv2.imread(sample_image_path)
                if original_img_color is not None:
                    original_img_resized = cv2.resize(original_img_color, (256, 256))
                    processed_sample_img = preprocess_image(sample_image_path, method=selected_pre_process_method)

                    if processed_sample_img is not None:
                        plt.figure(figsize=(12, 6))
                        plt.subplot(1, 2, 1)
                        plt.imshow(cv2.cvtColor(original_img_resized, cv2.COLOR_BGR2RGB))
                        plt.title("Citra Asli (Resized)")
                        plt.axis('off')

                        plt.subplot(1, 2, 2)
                        plt.imshow(processed_sample_img, cmap='gray')
                        plt.title(f"Citra Setelah {selected_pre_process_method.replace('_', ' ').title()}")
                        plt.axis('off')
                        plt.show()
                    else:
                        print(f"Tidak dapat memproses citra contoh: {sample_image_path}")
                else:
                    print(f"Tidak dapat membaca citra contoh asli: {sample_image_path}")
            else:
                print("Tidak ada citra di kelas pertama untuk visualisasi contoh.")
        else:
            print("Tidak ada kelas ditemukan dalam dataset untuk visualisasi contoh.")