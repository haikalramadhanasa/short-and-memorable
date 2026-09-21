import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_and_to_grayscale(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan di: {image_path}")

    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.astype(np.uint8)


def intensity_transformation(gray_img, alpha=1.2, beta=30):
    # Rumus: g(x,y) = alpha * f(x,y) + beta
    # alpha > 1 (peningkatan kontras), beta > 0 (peningkatan kecerahan)
    transformed = alpha * gray_img.astype(np.float64) + beta

    transformed = np.clip(transformed, 0, 255)
    return transformed.astype(np.uint8)


def histogram_equalization(gray_img):
    height, width = gray_img.shape
    total_pixels = height * width

    hist = np.zeros(256, dtype=int)
    for row in range(height):
        for col in range(width):
            pixel_val = gray_img[row, col]
            hist[pixel_val] += 1

    cdf = np.zeros(256, dtype=float)
    current_sum = 0
    for i in range(256):
        current_sum += hist[i]
        cdf[i] = current_sum

    # Rumus: h(v) = round( ((cdf(v) - cdf_min) / (total_pixels - cdf_min)) * (L - 1) )
    cdf_min = cdf[cdf > 0][0]  
    lookup_table = np.zeros(256, dtype=np.uint8)

    for i in range(256):
        if cdf[i] < cdf_min:
            lookup_table[i] = 0
        else:
            val = round(((cdf[i] - cdf_min) / (total_pixels - cdf_min)) * 255)
            lookup_table[i] = np.uint8(np.clip(val, 0, 255))

    equalized_img = np.zeros_like(gray_img)
    for row in range(height):
        for col in range(width):
            equalized_img[row, col] = lookup_table[gray_img[row, col]]

    return equalized_img, hist, lookup_table


path_gambar = "gambar.jpg"

try:
    gray_original = load_and_to_grayscale(path_gambar)

    img_transformed = intensity_transformation(
        gray_original, alpha=1.3, beta=20
    )

    img_equalized, hist_orig, lut = histogram_equalization(gray_original)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(gray_original, cmap="gray")
    plt.title("Citra Asli (Grayscale)")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(img_transformed, cmap="gray")
    plt.title("Transformasi Intensitas")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(img_equalized, cmap="gray")
    plt.title("Ekualisasi Histogram")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.hist(gray_original.ravel(), 256, [0, 256], color="black")
    plt.title("Hist. Asli")

    plt.subplot(2, 3, 5)
    plt.hist(img_transformed.ravel(), 256, [0, 256], color="blue")
    plt.title("Hist. Transformasi")

    plt.subplot(2, 3, 6)
    plt.hist(img_equalized.ravel(), 256, [0, 256], color="green")
    plt.title("Hist. Ekualisasi")

    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Error: {e}")