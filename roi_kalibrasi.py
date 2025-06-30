# Nama file: cari_koordinat_roi.py
import cv2
import numpy as np

points = []

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        cv2.circle(image, (x, y), 5, (0, 0, 255), 10)
        if len(points) > 1:
            cv2.line(image, tuple(points[-2]), tuple(points[-1]), (0, 255, 0), 10)
        cv2.imshow("Region of Interest (Pemilihan Area)", image)

image_path = r"D:\coding TA v2\data\CCTV\sample_4.png"
image = cv2.imread(image_path)
if image is None:
    print("Gambar tidak ditemukan")
    exit()

cv2.namedWindow("Region of Interest (Pemilihan Area)", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Region of Interest (Pemilihan Area)", click_event)

print("q untuk keluar")
cv2.imshow("Region of Interest (Pemilihan Area)", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

if points:
    print("\nHasil ROI Vertices':")
    print(f"np.array({points}, np.int32)")