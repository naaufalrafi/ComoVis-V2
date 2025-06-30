# Nama file: deteksi_final_v5_save_filters.py
import cv2
import numpy as np
import os

# --- PENGATURAN GLOBAL & PARAMETER TUNING ---
PIXELS_PER_METRIC = 2.835
MIN_DIM_CM = 50.0
MAX_DIM_CM = 150.0
roi_vertices = np.array([[1335, 327], [1200, 465], [1183, 608], [1217, 760], [1126, 896], 
[1122, 1025], [1126, 1283], [1160, 1398], [1402, 1477], [1604, 1483], 
[1828, 1493], [2024, 1490], [2070, 1375], [2096, 1302], [2129, 1209], 
[2163, 1126], [2155, 1046], [2203, 954], [2178, 726], [2089, 733], 
[1898, 590], [1771, 504], [1716, 438], [1666, 438], [1330, 322]], np.int32)

def deteksi_boulder(frame):
    output_frame = frame.copy()
    
    # Buat beberapa salinan frame untuk visualisasi setiap langkah filter
    viz_safe_zone = frame.copy()
    viz_area = frame.copy()
    viz_solidity = frame.copy()

    # (Langkah 1-5: ROI, CLAHE, HSV, Morfologi, Watershed - tidak berubah)
    # ...
    mask_roi = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask_roi, [roi_vertices], 255)
    safe_zone_kernel = np.ones((15, 15), np.uint8)
    safe_zone_mask = cv2.erode(mask_roi, safe_zone_kernel, iterations=1)
    roi_frame = cv2.bitwise_and(frame, frame, mask=mask_roi)
    hsv_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_frame)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_equalized = clahe.apply(v)
    hsv_equalized = cv2.merge([h, s, v_equalized])
    lower_bound = np.array([0, 0, 40])
    upper_bound = np.array([180, 80, 225])
    color_mask = cv2.inRange(hsv_equalized, lower_bound, upper_bound)
    kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel_closing)
    kernel_erosion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    separated_mask = cv2.erode(cleaned_mask, kernel_erosion, iterations=1)
    dist_transform = cv2.distanceTransform(separated_mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(separated_mask, np.ones((3, 3), np.uint8), iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(output_frame, markers)
    
    # Langkah 6: Pengukuran dan Penyaringan Hasil
    unique_markers = np.unique(markers)
    for marker_val in unique_markers:
        if marker_val <= 1: continue

        marker_mask_loop = np.zeros(frame.shape[:2], dtype="uint8")
        marker_mask_loop[markers == marker_val] = 255
        
        contours_ws, _ = cv2.findContours(marker_mask_loop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours_ws: continue
        
        cnt_ws = max(contours_ws, key=cv2.contourArea)
        
        # --- Proses Penyaringan Bertingkat ---
        # Filter 1: Zona Aman (Safe Zone)
        M = cv2.moments(cnt_ws)
        if M["m00"] == 0: continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        if safe_zone_mask[cY, cX] == 0: continue
        # Jika lolos, gambar di citra visualisasi Safe Zone
        cv2.drawContours(viz_safe_zone, [cnt_ws], -1, (0, 255, 255), 2) # Kuning

        # Filter 2: Luas Area dalam Piksel
        area = cv2.contourArea(cnt_ws)
        if area < 15000: continue
        # Jika lolos, gambar di citra visualisasi Area
        cv2.drawContours(viz_area, [cnt_ws], -1, (255, 0, 255), 2) # Magenta

        # Filter 3: Soliditas Bentuk
        hull = cv2.convexHull(cnt_ws)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: continue
        solidity = float(area) / hull_area
        if solidity < 0.70: continue
        # Jika lolos, gambar di citra visualisasi Soliditas
        cv2.drawContours(viz_solidity, [cnt_ws], -1, (255, 255, 0), 2) # Cyan

        # --- Akhir Proses Penyaringan ---

        # Jika lolos semua filter, lanjutkan proses pengukuran
        rect = cv2.minAreaRect(cnt_ws)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        (width_px, height_px) = rect[1]
        width_cm = width_px / PIXELS_PER_METRIC
        height_cm = height_px / PIXELS_PER_METRIC
        main_dimension = max(width_cm, height_cm)
        
        # Filter 4: Ukuran final dalam CM
        if MIN_DIM_CM <= main_dimension <= MAX_DIM_CM:
            cv2.drawContours(output_frame, [box], 0, (0, 255, 0), 3)
            # ... (kode putText tetap sama) ...
            
    # Kembalikan semua gambar yang kita butuhkan untuk analisis
    return output_frame, viz_safe_zone, viz_area, viz_solidity

# --- BAGIAN UTAMA UNTUK MENJALANKAN PENGUJIAN & MENYIMPAN HASIL FILTER ---
if __name__ == '__main__':
    folder_path = r"D:\coding TA v2\data\CCTV"
    list_nama_file = [
        "sample_1.png",
        "sample_2.png",
        "sample_3.png",
        "sample_4.png",
        "sample_5.png",
        "sample_6.png"
    ]

    print(f"Memulai pengujian untuk {len(list_nama_file)} gambar...")
    for nama_file in list_nama_file:
        image_path = os.path.join(folder_path, nama_file)
        print(f"\n--- Memproses: {image_path} ---")

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Gagal memuat gambar. Melanjutkan.")
            continue

        # Terima 4 gambar dari fungsi deteksi
        final_result, filtered_safe_zone, filtered_area, filtered_solidity = deteksi_boulder(image)

        # --- Simpan Hasil dari Setiap Filter ---
        nama_file_tanpa_ekstensi = os.path.splitext(nama_file)[0]
        
        # 1. Simpan hasil filter zona aman
        output_safezone = f"hasil_filter_safezone_{nama_file_tanpa_ekstensi}.jpg"
        cv2.imwrite(output_safezone, filtered_safe_zone)
        print(f"✅ Hasil Filter Zona Aman disimpan sebagai: {output_safezone}")
        
        # 2. Simpan hasil filter luas area
        output_area = f"hasil_filter_area_{nama_file_tanpa_ekstensi}.jpg"
        cv2.imwrite(output_area, filtered_area)
        print(f"✅ Hasil Filter Area disimpan sebagai: {output_area}")
        
        # 3. Simpan hasil filter soliditas
        output_solidity = f"hasil_filter_solidity_{nama_file_tanpa_ekstensi}.jpg"
        cv2.imwrite(output_solidity, filtered_solidity)
        print(f"✅ Hasil Filter Soliditas disimpan sebagai: {output_solidity}")
        
        # Tampilkan hanya hasil akhir
        window_name = f"HASIL AKHIR - {nama_file}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, final_result)
        
        print("Tekan tombol apa saja untuk lanjut, atau 'q' untuk keluar.")
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if key == ord('q'):
            print("Proses dihentikan oleh pengguna.")
            break
            
    print("\nPengujian selesai.")