import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("kamera tidak ditemukan")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    b, g, r = cv2.split(frame)
    zeros = np.zeros(frame.shape[:2], dtype="uint8")
    filtered = cv2.merge([zeros, zeros, r])

    cv2.imshow('Kamera Asli', frame)
    cv2.imshow('Kamera Filtered', filtered)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()