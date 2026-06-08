import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("output/steps", exist_ok=True)

img = cv2.imread("input/parking.jpg")

if img is None:
    print("Gambar tidak ditemukan")
    exit()

cv2.imwrite("output/steps/1_original.png", img)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
value = hsv[:, :, 2]

cv2.imwrite("output/steps/2_value_channel.png",value)

blur = cv2.GaussianBlur(value,(5, 5),0)

cv2.imwrite("output/steps/3_blur.png",blur)

kernel_top = cv2.getStructuringElement(cv2.MORPH_RECT,(15, 15))

top_hat = cv2.morphologyEx(blur,cv2.MORPH_TOPHAT,kernel_top)

cv2.imwrite("output/steps/4_top_hat.png",top_hat)

_, mask = cv2.threshold(top_hat,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imwrite("output/steps/5_threshold.png",mask)

kernel = np.ones((7, 7), np.uint8)
mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel,iterations=3)

mask = cv2.dilate(mask,kernel,iterations=1)

cv2.imwrite("output/steps/6_morphology.png",mask)

contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

hasil = img.copy()

jumlah_mobil = 0

MIN_AREA = 700
MAX_AREA = 10000

MIN_RATIO = 0.4
MAX_RATIO = 2.5

for cnt in contours:

    area = cv2.contourArea(cnt)

    if area < MIN_AREA:
        continue

    if area > MAX_AREA:
        continue

    x, y, w, h = cv2.boundingRect(cnt)

    ratio = w / float(h)

    if ratio < MIN_RATIO:
        continue

    if ratio > MAX_RATIO:
        continue

    jumlah_mobil += 1

    # perbesar bounding box
    padding = 20

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)

    x2 = min(img.shape[1], x + w + padding)
    y2 = min(img.shape[0], y + h + padding)

    cv2.rectangle(
        hasil,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    cv2.putText(hasil,str(jumlah_mobil),(x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255),2)

cv2.putText(
    hasil,
    f"Jumlah Mobil : {jumlah_mobil}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

cv2.imwrite(
    "output/result.png",
    hasil
)

print()
print(f"Jumlah Mobil Terdeteksi : {jumlah_mobil}")

gambar_step = [(img, "Original", "bgr"),(value, "Value Channel", "gray"),(blur, "Gaussian Blur", "gray"),(top_hat, "Top Hat", "gray"),(mask, "Morphology", "gray"),(hasil, "Result", "bgr")]

fig, ax = plt.subplots(2,3,figsize=(15, 8))

for a, (gambar, judul, mode) in zip(ax.flatten(),gambar_step):

    if mode == "bgr":
        a.imshow(cv2.cvtColor(gambar,cv2.COLOR_BGR2RGB))
    else:
        a.imshow(gambar,cmap="gray")

    a.set_title(judul)
    a.axis("off")

plt.tight_layout()

plt.savefig("output/steps/pipeline_visualization.png",dpi=120)

plt.show()