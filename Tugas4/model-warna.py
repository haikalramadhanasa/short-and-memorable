import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Baca citra
# ==========================================
path_gambar = 'gambar.jpg' 
img_bgr = cv2.imread(path_gambar)

if img_bgr is None:
    print("Error: Gambar tidak ditemukan. Periksa kembali path file!")
else:
    # konversi bgr ke rgb
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ==========================================
    # 2. Konversi ke HSV
    # ==========================================
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # ==========================================
    # 3. Konversi ke HSI
    # ==========================================
    def rgb_to_hsi(image_rgb):
        # Normalisasi
        rgb = image_rgb.astype(np.float32) / 255.0
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        
        # Intensity
        i = (r + g + b) / 3.0
        
        # Saturation
        min_rgb = np.minimum(np.minimum(r, g), b)
        s = np.zeros_like(i)
        sum_rgb = r + g + b
        mask = sum_rgb > 0
        s[mask] = 1.0 - (3.0 / sum_rgb[mask]) * min_rgb[mask]
        
        # Hue
        num = 0.5 * ((r - g) + (r - b))
        den = np.sqrt((r - g)**2 + (r - b) * (g - b))
        theta = np.arccos(np.clip(num / (den + 1e-6), -1.0, 1.0))
        
        h = np.zeros_like(i)
        h[b > g] = 2 * np.pi - theta[b > g]
        h[b <= g] = theta[b <= g]
        h = h / (2 * np.pi)  # Normalisasi hue
        
        hsi = np.stack([h, s, i], axis=-1)
        return (hsi * 255).astype(np.uint8)

    img_hsi = rgb_to_hsi(img_rgb)

    # ==========================================
    # 4. Konversi ke CMYK
    # ==========================================
    def rgb_to_cmyk(image_rgb):
        rgb = image_rgb.astype(np.float32) / 255.0
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        
        k = 1.0 - np.maximum(np.maximum(r, g), b)
        c = np.zeros_like(k)
        m = np.zeros_like(k)
        y = np.zeros_like(k)
        
        mask = k < 1.0
        c[mask] = (1.0 - r[mask] - k[mask]) / (1.0 - k[mask])
        m[mask] = (1.0 - g[mask] - k[mask]) / (1.0 - k[mask])
        y[mask] = (1.0 - b[mask] - k[mask]) / (1.0 - k[mask])
        
        cmyk = np.stack([c, m, y, k], axis=-1)
        return (cmyk * 255).astype(np.uint8)

    img_cmyk = rgb_to_cmyk(img_rgb)

    # ==========================================
    # 5. Visualisasi
    # ==========================================
    plt.figure(figsize=(14, 7))
    
    plt.subplot(2, 3, 1)
    plt.imshow(img_rgb)
    plt.title('RGB Original')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(img_hsv)
    plt.title('HSV')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(img_hsi)
    plt.title('HSI')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(img_cmyk[:, :, :3])
    plt.title('CMY Channels')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(img_cmyk[:, :, 3], cmap='gray')
    plt.title('Key / Black (K) Channel')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
