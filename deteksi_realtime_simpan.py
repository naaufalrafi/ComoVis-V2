import cv2
import numpy as np
import os
from datetime import datetime

# --- PENGATURAN GLOBAL & PARAMETER TUNING ---
PIXELS_PER_METRIC = 2.835
MIN_DIM_CM = 50.0
MAX_DIM_CM = 100.0
roi_vertices = np.array([[567, 69], [668, 111], [761, 146], [839, 176], [894, 251], [934, 297], [985, 317], [1020, 326], [1043, 366], [1029, 427], [1007, 479], [1002, 554], [989, 613], [943, 674], [951, 716], [697, 692], [542, 683], [466, 658], [413, 653], [412, 544], [462, 437], [496, 330], [518, 234], [534, 157], [565, 66]], np.int32)

def deteksi_boulder(frame):
    """
    Fungsi ini mengambil frame video dan melakukan deteksi batu di dalamnya.
    """
    output_frame = frame.copy()
    
    # Membuat mask untuk Region of Interest (ROI)
    mask_roi = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask_roi, [roi_vertices], 255)
    
    # Membuat zona aman yang sedikit lebih kecil dari ROI untuk menghindari deteksi di tepi
    safe_zone_kernel = np.ones((15, 15), np.uint8)
    safe_zone_mask = cv2.erode(mask_roi, safe_zone_kernel, iterations=1)
    
    # Pre-processing gambar
    roi_frame = cv2.bitwise_and(frame, frame, mask=mask_roi)
    hsv_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_frame)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_equalized = clahe.apply(v)
    hsv_equalized = cv2.merge([h, s, v_equalized])
    
    # Segmentasi warna
    lower_bound = np.array([0, 0, 40])
    upper_bound = np.array([180, 80, 225])
    color_mask = cv2.inRange(hsv_equalized, lower_bound, upper_bound)
    
    # Operasi morfologi untuk membersihkan mask
    kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel_closing)
    kernel_erosion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    separated_mask = cv2.erode(cleaned_mask, kernel_erosion, iterations=1)
    
    # Algoritma Watershed untuk memisahkan objek yang bersentuhan
    dist_transform = cv2.distanceTransform(separated_mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(separated_mask, np.ones((3, 3), np.uint8), iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(output_frame, markers)
    
    # Pengukuran dan penyaringan hasil dari Watershed
    unique_markers = np.unique(markers)
    for marker_val in unique_markers:
        if marker_val <= 1:
            continue
        marker_mask_loop = np.zeros(frame.shape[:2], dtype="uint8")
        marker_mask_loop[markers == marker_val] = 255
        
        contours_ws, _ = cv2.findContours(marker_mask_loop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours_ws:
            continue
        
        cnt_ws = max(contours_ws, key=cv2.contourArea)
        area = cv2.contourArea(cnt_ws)
        
        # Filter berdasarkan area, zona aman, dan soliditas
        if area < 8000:
            continue
            
        M = cv2.moments(cnt_ws)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        if safe_zone_mask[cY, cX] == 0:
            continue

        hull = cv2.convexHull(cnt_ws)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = float(area) / hull_area
        if solidity < 0.65:
            continue

        # Pengukuran dimensi dan visualisasi hasil
        rect = cv2.minAreaRect(cnt_ws)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        (width_px, height_px) = rect[1]
        width_cm = width_px / PIXELS_PER_METRIC
        height_cm = height_px / PIXELS_PER_METRIC
        main_dimension = max(width_cm, height_cm)
        
        if MIN_DIM_CM <= main_dimension <= MAX_DIM_CM:
            cv2.drawContours(output_frame, [box], 0, (0, 255, 0), 3)
            label_text = f"{main_dimension:.1f} cm"
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(output_frame, (cX - 55, cY - text_height), (cX + text_width - 50, cY + 5), (0, 0, 0), -1)
            cv2.putText(output_frame, label_text, (cX - 50, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
    return output_frame

# --- BAGIAN UTAMA UNTUK MENJALANKAN DETEKSI REAL-TIME ---
if __name__ == '__main__':
    # --- GANTI SUMBER VIDEO DI SINI ---
    # Opsi 1: Untuk webcam/CCTV via USB (0 adalah kamera default)
    video_source = 0
    
    # Opsi 2: Untuk IP Camera (ganti dengan URL RTSP Anda)
    # video_source = "rtsp://username:password@192.168.1.10:554/stream1"
    
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Tidak bisa membuka sumber video '{video_source}'")
        exit()

    print("Sumber video berhasil dibuka. Menjalankan deteksi...")
    print("--- Tekan 's' untuk MULAI/BERHENTI merekam video ---")
    print("--- Tekan 'q' untuk KELUAR ---")

    # Inisialisasi variabel untuk fungsionalitas perekaman
    is_saving = False
    out = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Tidak dapat menerima frame dari sumber video. Program berhenti.")
            break

        # Lakukan proses deteksi pada setiap frame
        final_result = deteksi_boulder(frame)
        
        # Tambahkan indikator visual "MEREKAM" jika status penyimpanan aktif
        if is_saving:
            cv2.putText(final_result, "MEREKAM", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Tampilkan hasil deteksi ke layar
        cv2.namedWindow("Deteksi Boulder Real-time", cv2.WINDOW_NORMAL)
        cv2.imshow("Deteksi Boulder Real-time", final_result)
        
        # Tangkap input keyboard
        key = cv2.waitKey(1) & 0xFF

        # Logika tombol 'q' untuk keluar
        if key == ord('q'):
            print("Tombol 'q' ditekan, program akan keluar.")
            break
        
        # Logika tombol 's' untuk memulai atau menghentikan perekaman
        if key == ord('s'):
            is_saving = not is_saving  # Balik status perekaman

            if is_saving:
                # Jika mulai merekam, siapkan file video output
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"hasil_deteksi_{timestamp}.mp4"
                
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps == 0:
                    fps = 20.0  # Nilai default jika FPS tidak terdeteksi
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
                print(f"--- MULAI MEREKAM --- Hasil disimpan ke: {output_filename}")
            else:
                # Jika berhenti merekam, lepaskan file video
                if out is not None:
                    out.release()
                    out = None
                    print("--- PEREKAMAN DIHENTIKAN ---")

        # Jika status 'is_saving' aktif, tulis frame ke file video
        if is_saving and out is not None:
            out.write(final_result)
            
    # Pastikan semua resource dilepaskan saat keluar dari loop
    if out is not None:
        out.release()
        
    cap.release()
    cv2.destroyAllWindows()
    print("Program selesai dan semua jendela telah ditutup.")