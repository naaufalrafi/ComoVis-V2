# Nama file: deteksi_final_v5_save_segmentation.py
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

    # (Langkah 1 & 2: ROI dan CLAHE - tidak berubah)
    mask_roi = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask_roi, [roi_vertices], 255)
    safe_zone_kernel = np.ones((15,15), np.uint8)
    safe_zone_mask = cv2.erode(mask_roi, safe_zone_kernel, iterations=1)
    roi_frame = cv2.bitwise_and(frame, frame, mask=mask_roi)
    hsv_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_frame)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_equalized = clahe.apply(v)
    hsv_equalized = cv2.merge([h, s, v_equalized])

    # Langkah 3: Segmentasi Warna (HSV)
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
    
    # Buat visualisasi hasil watershed untuk disimpan
    watershed_visualization = np.zeros_like(frame, dtype=np.uint8)
    watershed_visualization[markers == -1] = [0, 0, 255] # Garis batas berwarna merah

    # (Langkah 6: Pengukuran dan Penyaringan Hasil - tidak berubah)
    unique_markers = np.unique(markers)
    # ... (Sisa kode pengukuran dan filter area, soliditas, dll tetap sama) ...
            
    # --- PERUBAHAN PADA RETURN STATEMENT ---
    # Kembalikan semua gambar intermediate yang kita butuhkan
    return output_frame, color_mask, cleaned_mask, watershed_visualization


# --- BAGIAN UTAMA UNTUK MENJALANKAN PENGUJIAN & MENYIMPAN HASIL ---
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

        # --- PERUBAHAN PADA PEMANGGILAN FUNGSI ---
        # Sekarang kita menerima 4 nilai kembali
        final_result, mask_awal, mask_dibersihkan, watershed_viz = deteksi_boulder(image)

        # --- PENAMBAHAN KODE UNTUK MENYIMPAN HASIL SEGMENTASI ---
        nama_file_tanpa_ekstensi = os.path.splitext(nama_file)[0]
        
        # 1. Simpan masker warna awal
        output_mask_awal = f"hasil_masker_warna_{nama_file_tanpa_ekstensi}.jpg"
        cv2.imwrite(output_mask_awal, mask_awal)
        print(f"✅ Masker warna awal disimpan sebagai: {output_mask_awal}")
        
        # 2. Simpan masker setelah dibersihkan
        output_mask_bersih = f"hasil_masker_dibersihkan_{nama_file_tanpa_ekstensi}.jpg"
        cv2.imwrite(output_mask_bersih, mask_dibersihkan)
        print(f"✅ Masker yang dibersihkan disimpan sebagai: {output_mask_bersih}")
        
        # 3. Simpan visualisasi watershed
        output_watershed = f"hasil_watershed_{nama_file_tanpa_ekstensi}.jpg"
        cv2.imwrite(output_watershed, watershed_viz)
        print(f"✅ Hasil watershed disimpan sebagai: {output_watershed}")
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