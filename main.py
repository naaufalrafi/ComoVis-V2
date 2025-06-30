# Nama file: deteksi_lanjutan.py
import cv2
import numpy as np

# --- PENGATURAN AWAL ---
PIXELS_PER_METRIC = 2.835  # Rasio kalibrasi Anda
MIN_DIM_CM = 50.0
MAX_DIM_CM = 150.0

# --- PENTING: TENTUKAN REGION OF INTEREST (ROI) ANDA DI SINI ---
# Ganti koordinat ini dengan hasil dari script 'cari_koordinat_roi.py'
# Format: np.array([[x1, y1], [x2, y2], ...], np.int32)
roi_vertices = np.array([[1157, 288], [1172, 551], [1162, 744], [1092, 967], [964, 1069], [867, 1207], [923, 1311], [1214, 1391], [1575, 1442], 
[1912, 1477], [2047, 1477], [2070, 1327], [2110, 1308], [2167, 786], [1957, 647], [1723, 546], [1531, 440], [1160, 290]], np.int32)


def process_image_advanced(frame):
    """
    Fungsi deteksi canggih menggunakan ROI Masking dan Watershed Algorithm.
    VERSI DEBUG: Menampilkan semua deteksi tanpa filter ukuran min/max.
    """
    original_frame = frame.copy()

    # 1. TERAPKAN REGION OF INTEREST (ROI)
    mask = np.zeros_like(original_frame)
    cv2.fillPoly(mask, [roi_vertices], (255, 255, 255))
    roi_frame = cv2.bitwise_and(original_frame, mask)

    # Pra-pemrosesan pada gambar ROI
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 2. PERSIAPAN UNTUK WATERSHED
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # 3. JALANKAN ALGORITMA WATERSHED
    markers = cv2.watershed(original_frame, markers)
    
    # 4. UKUR SETIAP SEGMEN BATUAN DAN TAMPILKAN SEMUA
    unique_markers = np.unique(markers)
    for marker_val in unique_markers:
        if marker_val <= 1:
            continue

        marker_mask = np.zeros(gray.shape, dtype="uint8")
        marker_mask[markers == marker_val] = 255
        
        contours, _ = cv2.findContours(marker_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        
        cnt = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(cnt) < 200:
            continue

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        (width_px, height_px) = rect[1]
        width_cm = width_px / PIXELS_PER_METRIC
        height_cm = height_px / PIXELS_PER_METRIC
        main_dimension = max(width_cm, height_cm)
        
        # --- PERUBAHAN UTAMA ADA DI SINI ---
        
        # 1. Tentukan warna dan label berdasarkan ukuran
        if MIN_DIM_CM <= main_dimension <= MAX_DIM_CM:
            color = (0, 255, 0)  # Hijau
            label = "STANDAR"
        else:
            color = (0, 0, 255)  # Merah
            label = "TIDAK STANDAR"
            
        # 2. Gambar SEMUA kotak deteksi, tanpa mempedulikan ukurannya.
        # Blok if untuk menyaring telah dihapus dari sekeliling kode gambar.
        cv2.drawContours(original_frame, [box], 0, color, 2)
        label_text = f"{label}: {main_dimension:.1f} cm"
        cv2.putText(original_frame, label_text, (box[1][0], box[1][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        # --- AKHIR PERUBAHAN ---

    return original_frame

# --- BAGIAN UTAMA UNTUK MENGUJI GAMBAR ---
if __name__ == '__main__':
    image_path = r"D:\coding TA v2\data\CCTV\sample_1.png"
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Gagal memuat gambar dari {image_path}")
    else:
        result_image = process_image_advanced(image)
        cv2.namedWindow("Hasil Deteksi Lanjutan", cv2.WINDOW_NORMAL)
        cv2.imshow("Hasil Deteksi Lanjutan", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()