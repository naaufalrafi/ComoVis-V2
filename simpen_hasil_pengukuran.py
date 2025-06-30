# ===================================================================
# NAMA FILE: deteksi_boulder_final_production.py
# DESKRIPSI: Versi produksi final untuk deteksi dan pengukuran boulder.
#            Hanya menyimpan gambar hasil akhir.
# ===================================================================

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
    """
    Fungsi utama untuk mendeteksi boulder pada sebuah gambar (frame).
    """
    output_frame = frame.copy()

    # Langkah 1: Membuat Masker ROI dan "Safe Zone"
    mask_roi = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask_roi, [roi_vertices], 255)
    safe_zone_kernel = np.ones((15, 15), np.uint8)
    safe_zone_mask = cv2.erode(mask_roi, safe_zone_kernel, iterations=1)
    roi_frame = cv2.bitwise_and(frame, frame, mask=mask_roi)
    
    # Langkah 2: Normalisasi Kontras CLAHE
    hsv_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_frame)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_equalized = clahe.apply(v)
    hsv_equalized = cv2.merge([h, s, v_equalized])

    # Langkah 3: Segmentasi Warna
    lower_bound = np.array([0, 0, 40])
    upper_bound = np.array([180, 80, 225])
    color_mask = cv2.inRange(hsv_equalized, lower_bound, upper_bound)
    
    # Langkah 4: Pembersihan dan Pemisahan Objek dengan Morfologi
    kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel_closing)
    kernel_erosion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    separated_mask = cv2.erode(cleaned_mask, kernel_erosion, iterations=1)

    # Langkah 5: Segmentasi Lanjutan dengan Watershed
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
        area = cv2.contourArea(cnt_ws)
        
        # Filter 1: Area dalam Piksel
        if area < 15000: continue
        
        # Filter 2: Zona Aman
        M = cv2.moments(cnt_ws)
        if M["m00"] == 0: continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        if safe_zone_mask[cY, cX] == 0: continue

        # Filter 3: Soliditas Bentuk
        hull = cv2.convexHull(cnt_ws)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: continue
        solidity = float(area) / hull_area
        if solidity < 0.70: continue

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
            label_text = f"{main_dimension:.1f} cm"
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(output_frame, (cX - 55, cY - text_height), (cX + text_width - 50, cY + 5), (0,0,0), -1)
            cv2.putText(output_frame, label_text, (cX - 50, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
    # Kembalikan hanya gambar hasil akhir
    return output_frame


# --- BAGIAN UTAMA UNTUK MENJALANKAN PENGUJIAN & MENYIMPAN HASIL AKHIR ---
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

        # Terima satu gambar hasil akhir dari fungsi deteksi
        final_result = deteksi_boulder(image)

        # --- Simpan Hanya Hasil Pengukuran Final ---
        nama_file_tanpa_ekstensi = os.path.splitext(nama_file)[0]
        output_filename = f"HASIL_PENGUKURAN_{nama_file_tanpa_ekstensi}.jpg"
        
        cv2.imwrite(output_filename, final_result)
        print(f"✅ Hasil Pengukuran Final disimpan sebagai: {output_filename}")
        # --- AKHIR PENAMBAHAN KODE ---

        # Tampilkan hasil akhir
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