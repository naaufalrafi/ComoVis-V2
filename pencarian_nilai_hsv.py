# Nama file: cari_nilai_hsv.py
import cv2
import numpy as np

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Dapatkan warna BGR dari piksel yang diklik
        bgr_color = image[y, x]
        
        # Ubah BGR ke HSV. Perhatikan kita perlu membuat array 3D kecil untuk konversi.
        hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]
        
        print(f"Koordinat (x,y): ({x}, {y}) | Warna BGR: {bgr_color} | Warna HSV: {hsv_color}")

# --- GANTI PATH GAMBAR DI SINI ---
# Gunakan sample_1 yang asli, bukan yang sudah ada kotaknya
image_path = r"D:\coding TA v2\data\CCTV\sample_1.png"

image = cv2.imread(image_path)
if image is None:
    print(f"Error: Gagal memuat gambar dari {image_path}")
    exit()

# Tampilkan gambar di jendela yang bisa di-resize
cv2.namedWindow("Klik pada Batu Target", cv2.WINDOW_NORMAL)
cv2.imshow("Klik pada Batu Target", image)

# Setel fungsi callback mouse
cv2.setMouseCallback("Klik pada Batu Target", click_event)

print("--- Petunjuk ---")
print("Klik pada beberapa titik di atas batu-batu target Anda.")
print("Perhatikan nilai HSV yang tercetak di terminal.")
print("Tekan 'q' untuk keluar.")
print("----------------")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()