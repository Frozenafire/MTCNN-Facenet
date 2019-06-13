import cv2
import os


path = 'C:/Users/xuefe/Desktop/facex/a/'
face_dirs = os.listdir(path)


def lap(img):
    score = cv2.Laplacian(img, cv2.CV_64F).var()
    return score


for im in face_dirs:
    img = path+im
    img = cv2.imread(img)
    s = lap(img)
    if s <= 20:
        cv2.imwrite('C:/Users/xuefe/Desktop/facex/face1/'+im, img)
    if (s > 20) and (s <= 100):
        cv2.imwrite('C:/Users/xuefe/Desktop/facex/face2/'+im, img)
    if s > 100:
        cv2.imwrite('C:/Users/xuefe/Desktop/facex/face3/'+im, img)
