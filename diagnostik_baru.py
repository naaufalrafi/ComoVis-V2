# Nama file: deteksi_boulder_final.py
import cv2
import numpy as np

# --- PENGATURAN AWAL ---
PIXELS_PER_METRIC = 2.835
MIN_DIM_CM = 50.0
MAX_DIM_CM = 150.0

roi_vertices = np.array([[1335, 327], [1200, 465], [1183, 608], [1217, 760], [1126, 896], [1122, 1025], [1126, 1283], 
[1160, 1398], [1402, 1477], [1604, 1483], [1828, 1493], [2024, 1490], [2070, 1375], [2096, 1302], [2129, 1209], [2163, 1126], [2155, 1046], 
[2203, 954], [2178, 726], [2089, 733], [1898, 590], [1771, 504], [1716, 438], [1666, 438], [1330, 322]], np.int32)

def deteksi_batuan(frame):
    output_frame = frame.copy()

    # 1. Terapkan ROI
    mask_roi = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask_roi, [roi_vertices], 255)
    roi_frame = cv2.bitwise_and(frame, frame, mask=mask_roi)

    # 2. Segmentasi Warna
    hsv_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 0, 100])
    upper_bound = np.array([180, 60, 220])
    color_mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    
    # 3. Watershed
    dist_transform = cv2.distanceTransform(cleaned_mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    sure_bg = cv2.dilate(cleaned_mask, np.ones((3,3),np.uint8), iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(output_frame, markers)
    
    # 4. UKUR & GAMBAR HASIL DENGAN FILTER BARU
    unique_markers = np.unique(markers)
    for marker_val in unique_markers:
        if marker_val <= 1: continue

        marker_mask = np.zeros(frame.shape[:2], dtype="uint8")
        marker_mask[markers == marker_val] = 255
        
        contours_ws, _ = cv2.findContours(marker_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours_ws: continue
        
        cnt_ws = max(contours_ws, key=cv2.contourArea)
        
        # --- PENYEMPURNAAN FINAL DI SINI ---
        # Naikkan nilai ini untuk memfilter deteksi yang tidak diinginkan
        if cv2.contourArea(cnt_ws) < 3000: 
            continue
        
        rect = cv2.minAreaRect(cnt_ws)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        (width_px, height_px) = rect[1]
        width_cm = width_px / PIXELS_PER_METRIC
        height_cm = height_px / PIXELS_PER_METRIC
        main_dimension = max(width_cm, height_cm)
        
        if MIN_DIM_CM <= main_dimension <= MAX_DIM_CM:
            cv2.drawContours(output_frame, [box], 0, (0, 255, 0), 3)
            label_text = f"{main_dimension:.1f} cm"
            cv2.putText(output_frame, label_text, (int(rect[0][0]), int(rect[0][1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
            cv2.putText(output_frame, label_text, (int(rect[0][0]), int(rect[0][1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
    return output_frame

# --- BAGIAN UTAMA ---
if __name__ == '__main__':
    image_path = r"D:\coding TA v2\data\CCTV\sample_1.png"
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Gagal memuat gambar dari {image_path}")
    else:
        result_image = deteksi_batuan(image)
        cv2.namedWindow("HASIL AKHIR", cv2.WINDOW_NORMAL)
        cv2.imshow("HASIL AKHIR", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()