import numpy as np
import cv2
import os


path = 'C:/Users/xuefe/Desktop/facex/face1/'
face_dirs = os.listdir(path)
for im in face_dirs:
    img = path + im
    img = cv2.imread(img)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
    img = cv2.filter2D(img, -1, kernel=kernel)
    cv2.imwrite('C:/Users/xuefe/Desktop/facex/face11/' + im, img)
