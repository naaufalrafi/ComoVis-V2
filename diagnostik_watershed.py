# Nama file: diagnostik_watershed.py
import cv2
import numpy as np

# --- PENGATURAN AWAL ---
# (Pastikan semua pengaturan ini sama dengan script utama Anda)
PIXELS_PER_METRIC = 2.835
roi_vertices = np.array([[1157, 288], [1172, 551], [1162, 744], [1092, 967], [964, 1069], [867, 1207], [923, 1311], [1214, 1391], [1575, 1442], 
[1912, 1477], [2047, 1477], [2070, 1327], [2110, 1308], [2167, 786], [1957, 647], [1723, 546], [1531, 440], [1160, 290]], np.int32)

def run_diagnostics(frame):
    """
    Fungsi ini akan menjalankan proses dan menampilkan setiap langkahnya.
    """
    # 1. Tampilkan ROI
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, [roi_vertices], (255, 255, 255))
    roi_frame = cv2.bitwise_and(frame, mask)
    cv2.imshow("1. Hasil ROI", roi_frame)

    # 2. Tampilkan Hasil Threshold
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow("2. Hasil Threshold Biner", thresh)

    # 3. Tampilkan Hasil "Sure Foreground" (Paling Penting!)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    
    # Coba ubah nilai 0.2 ini jika jendela 'Sure Foreground' masih hitam
    _, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    cv2.imshow("3. Sure Foreground (INTI BATUAN)", sure_fg)
    
    # 4. Tampilkan "Sure Background"
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    cv2.imshow("4. Sure Background", sure_bg)

    # 5. Tampilkan Area "Unknown"
    unknown = cv2.subtract(sure_bg, sure_fg)
    cv2.imshow("5. Area Tidak Diketahui", unknown)
    
    print("\n--- HASIL DIAGNOSTIK ---")
    print("Periksa semua jendela yang muncul, terutama '3. Sure Foreground'.")
    print("Apakah di jendela nomor 3 muncul bercak-bercak putih di tengah batuan?")
    print("Jika jendela nomor 3 hitam total, berarti nilai threshold watershed (0.2) masih terlalu tinggi.")
    print("Tekan tombol apa saja di keyboard untuk menutup semua jendela.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --- BAGIAN UTAMA UNTUK MENJALANKAN DIAGNOSTIK ---
if __name__ == '__main__':
    image_path = r"D:\coding TA v2\data\CCTV\sample_1.png"
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Gagal memuat gambar dari {image_path}")
    else:
        # Atur agar semua jendela bisa di-resize
        cv2.namedWindow("1. Hasil ROI", cv2.WINDOW_NORMAL)
        cv2.namedWindow("2. Hasil Threshold Biner", cv2.WINDOW_NORMAL)
        cv2.namedWindow("3. Sure Foreground (INTI BATUAN)", cv2.WINDOW_NORMAL)
        cv2.namedWindow("4. Sure Background", cv2.WINDOW_NORMAL)
        cv2.namedWindow("5. Area Tidak Diketahui", cv2.WINDOW_NORMAL)
        
        run_diagnostics(image)