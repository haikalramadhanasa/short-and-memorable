import cv2
import numpy as np
import matplotlib.pyplot as plt


img_bgr = cv2.imread('gambar.jpg')

if img_bgr is None:
    raise FileNotFoundError("Gambar tidak ditemukan. Periksa kembali lokasi file!")

# Konversi BGR (format OpenCV) ke RGB (format Matplotlib)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# A. Mean Filter
mean_filtered = cv2.blur(img_rgb, (25, 25))

# B. Gaussian Filter 
gaussian_filtered = cv2.GaussianBlur(img_rgb, (25, 25), 0)

# C. Median Filter
median_filtered = cv2.medianBlur(img_rgb, 25)

# D. Sharpening Filter 
kernel_sharpen = np.array([[ 0, -1,  0],
                           [-1,  5, -1],
                           [ 0, -1,  0]])
sharpened = cv2.filter2D(img_rgb, -1, kernel_sharpen)

# Tampilkan hasil
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle("Penerapan Filter Spasial Citra Digital", fontsize=14)

images = [
    (img_rgb, "Asli"),
    (mean_filtered, "Mean Filter (5x5)"),
    (gaussian_filtered, "Gaussian Filter (5x5)"),
    (median_filtered, "Median Filter (5x5)"),
    (sharpened, "Sharpening (Penajaman)")
]

for i, (img, title) in enumerate(images):
    row, col = divmod(i, 3)
    axes[row, col].imshow(img)
    axes[row, col].set_title(title)
    axes[row, col].axis('off')

# Sembunyikan subplot sisa (jika ada)
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()
