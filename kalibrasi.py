import cv2
import math

# List untuk menyimpan koordinat dua titik
ref_points = []

def click_event(event, x, y, flags, params):
    """Fungsi callback untuk menangani klik mouse."""
    global ref_points

    # Jika mouse kiri diklik
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(ref_points) < 2:
            ref_points.append((x, y))
            # Gambar lingkaran di titik yang diklik
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Ukur Piksel", image)

# Muat gambar yang sudah kita simpan
image_path = r"D:\coding TA v2\data\CCTV\sample_1.png"

# Muat gambar dari path yang sudah ditentukan
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Gagal memuat gambar dari path: {image_path}")
    print("Pastikan path sudah benar dan file gambar tidak rusak.")
    exit()

clone = image.copy()

cv2.namedWindow("Ukur Piksel", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Ukur Piksel", click_event)

print("--- Petunjuk ---")
print("1. Klik di ujung kiri objek referensi Anda.")
print("2. Klik di ujung kanan objek referensi Anda.")
print("3. Tekan tombol 'r' untuk reset jika salah klik.")
print("4. Tekan tombol 'q' untuk keluar.")
print("----------------")

while True:
    cv2.imshow("Ukur Piksel", image)
    key = cv2.waitKey(1) & 0xFF

    # Jika sudah ada dua titik, gambar garis dan hitung jarak
    if len(ref_points) == 2:
        cv2.line(image, ref_points[0], ref_points[1], (0, 0, 255), 2)

        # Hitung jarak Euclidean
        pixel_distance = math.sqrt((ref_points[1][0] - ref_points[0][0])**2 + (ref_points[1][1] - ref_points[0][1])**2)

        print(f"\nJarak terukur: {pixel_distance:.2f} piksel")

        # Tampilkan jarak pada gambar
        mid_point = ((ref_points[0][0] + ref_points[1][0]) // 2, (ref_points[0][1] + ref_points[1][1]) // 2)
        cv2.putText(image, f"{pixel_distance:.2f} px", (mid_point[0] - 30, mid_point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Ukur Piksel", image)
        ref_points.append(None) # Hentikan perhitungan ulang

    # Tombol 'r' untuk reset
    if key == ord('r'):
        image = clone.copy()
        ref_points = []
        print("Reset. Silakan klik lagi.")

    # Tombol 'q' untuk keluar
    elif key == ord('q'):
        break

cv2.destroyAllWindows()