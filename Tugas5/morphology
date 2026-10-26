import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Baca citra dalam mode Grayscale 
img = cv2.imread('gambar.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("path not found")
else:
    
    # 2. Binarisasi

    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    # 3. Definisikan Kernel (Structuring Element)
    kernel = np.ones((5, 5), np.uint8)

    # 4. Implementasi Operasi Morfologi
    
    # A. Erosi (Memperkecil objek putih)
    erosion = cv2.erode(thresh, kernel, iterations=1)

    # B. Dilasi (Memperbesar objek putih)
    dilation = cv2.dilate(thresh, kernel, iterations=1)

    # C. Opening: Erosi lalu Dilasi (menghilangkan noise/bintik putih di luar objek)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # D. Closing: Dilasi lalu Erosi (menutup lubang kecil di dalam objek)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 5. Visualisasi Hasil Menggunakan Matplotlib
    titles = ['Citra Asli (Thresh)', 'Erosi', 'Dilasi', 'Opening', 'Closing']
    images = [thresh, erosion, dilation, opening, closing]

    plt.figure(figsize=(12, 6))
    for i in range(5):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()
