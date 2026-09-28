# Tugas 3: Penerapan Filter Spasial pada Citra Digital

Dokumentasi dan implementasi filter spasial (*spatial filtering*) menggunakan Python, OpenCV, dan Matplotlib untuk perataan (*smoothing*) serta penajaman (*sharpening*) citra.

---

## Deskripsi Tugas
Tugas ini bertujuan untuk menerapkan beberapa teknik filter spasial pada citra digital berbasis matriks konvolusi (kernel). Filter spasial yang diimplementasikan meliputi:

1. **Mean Filter (Averaging)**: Meredam noise acak dengan mengambil rata-rata piksel tetangga.
2. **Gaussian Filter**: Menghaluskan citra secara lebih alami dengan bobot Gaussian.
3. **Median Filter**: Filter non-linier yang sangat efektif menghilangkan noise *Salt & Pepper*.
4. **Sharpening Filter**: Menajamkan garis tepi (*edges*) dan detail citra menggunakan kernel *high-pass*.

## Hasil filter

![Hasil Filter Spasial](gambar.jpg)
