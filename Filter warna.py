import cv2
import numpy as np

img = cv2.imread('picturrrre.jpg')

if img is None:
    print("gambar hilang")
else:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    b, g, r = cv2.split(img)
    zeros = np.zeros(img.shape[:2], dtype="uint8")
    
    yellow_filter = cv2.merge([zeros, g, r])
    
    cyan_filter = cv2.merge([b, g, zeros])
    
    random_filter = cv2.merge([g, zeros, r])

    blue_filter = cv2.merge([b, zeros, zeros])

    green_filter = cv2.merge([zeros, g, zeros])

    red_filter = cv2.merge([zeros, zeros, r])

    # hasil
    cv2.imshow('Asli', img)
    cv2.imshow('grey', gray)
    cv2.imshow('red', red_filter)
    cv2.imshow('gfeen', green_filter)
    cv2.imshow('blue', blue_filter)
    cv2.imshow('kuning', yellow_filter)
    cv2.imshow('birulagi', cyan_filter)
    cv2.imshow('random', random_filter)

    cv2.waitKey(0)
    cv2.destroyAllWindows()