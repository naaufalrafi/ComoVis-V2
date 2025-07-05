import cv2
import numpy as np
import os

# --- PENGATURAN GLOBAL & PARAMETER TUNING ---
DEBUG_MODE = True # Biarkan True untuk memantau hasil

PIXELS_PER_METRIC = 2.835
MIN_DIM_CM = 50.0
MAX_DIM_CM = 100.0
roi_vertices = np.array([[567, 69], [668, 111], [761, 146], [839, 176], [894, 251], [934, 297], [985, 317], [1020, 326], [1043, 366], [1029, 427], [1007, 479], [1002, 554], [989, 613], [943, 674], [951, 716], [697, 692], [542, 683], [466, 658], [413, 653], [412, 544], [462, 437], [496, 330], [518, 234], [534, 157], [565, 66]], np.int32)

def deteksi_boulder(frame):
    output_frame = frame.copy()

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

    # === PERUBAHAN 1: Sedikit melonggarkan HSV untuk bayangan ===
    # Kembali ke parameter awal, hanya ubah V-min untuk menangani bayangan.
    lower_bound = np.array([0, 0, 30])      # <-- PERUBAHAN (dari 40 ke 30)
    upper_bound = np.array([180, 80, 225]) # <-- KEMBALI KE ASLI
    color_mask = cv2.inRange(hsv_equalized, lower_bound, upper_bound)
    
    kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel_closing)
    
    # === PERUBAHAN 2: Mengurangi erosi agar tidak memecah batuan besar ===
    kernel_erosion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) # <-- PERUBAHAN (dari 5,5 ke 3,3)
    separated_mask = cv2.erode(cleaned_mask, kernel_erosion, iterations=1)
    
    dist_transform = cv2.distanceTransform(separated_mask, cv2.DIST_L2, 5)

    # === PERUBAHAN 3 (KUNCI): Kembalikan ambang batas ke nilai tinggi dan stabil ===
    # Kita gunakan 0.5 untuk memastikan hanya inti paling jelas yang jadi 'benih'
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0) # <-- PERUBAHAN KUNCI
    sure_fg = np.uint8(sure_fg)
    
    # ... (sisa kode watershed sama)
    sure_bg = cv2.dilate(separated_mask, np.ones((3, 3), np.uint8), iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(output_frame, markers)

    if DEBUG_MODE:
        dist_display = cv2.normalize(dist_transform, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imshow("1. Mask Awal (HSV)", color_mask)
        cv2.imshow("2. Mask Setelah Morfologi", separated_mask)
        cv2.imshow("3. Transformasi Jarak", dist_display)
        cv2.imshow("4. Inti Objek (Sure FG)", sure_fg)
    
    unique_markers = np.unique(markers)
    for marker_val in unique_markers:
        if marker_val <= 1: continue
        marker_mask_loop = np.zeros(frame.shape[:2], dtype="uint8")
        marker_mask_loop[markers == marker_val] = 255
        
        contours_ws, _ = cv2.findContours(marker_mask_loop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours_ws: continue
        
        cnt_ws = max(contours_ws, key=cv2.contourArea)
        area = cv2.contourArea(cnt_ws)
        
        # Kembali ke filter area awal Anda
        if area < 8000: continue # <-- KEMBALI KE ASLI
        
        M = cv2.moments(cnt_ws)
        if M["m00"] == 0: continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        if safe_zone_mask[cY, cX] == 0: continue

        hull = cv2.convexHull(cnt_ws)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: continue
        solidity = float(area) / hull_area
        # Kembali ke filter soliditas awal
        if solidity < 0.65: continue # <-- KEMBALI KE ASLI

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
            cv2.rectangle(output_frame, (cX - 55, cY - text_height), (cX + text_width - 50, cY + 5), (0,0,0), -1)
            cv2.putText(output_frame, label_text, (cX - 50, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
    return output_frame

# --- BAGIAN UTAMA ---
if __name__ == '__main__':
    video_source = r"D:\coding TA v2\data\video\data_JI.mp4" 
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Tidak bisa membuka file video '{video_source}'")
        exit()

    print(f"Memproses video: {video_source}... Tekan 'q' pada jendela video untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video selesai, mengulang dari awal...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        final_result = deteksi_boulder(frame)
        cv2.namedWindow("Deteksi Boulder Real-time", cv2.WINDOW_NORMAL)
        cv2.imshow("Deteksi Boulder Real-time", final_result)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("Program dihentikan.")