import cv2

video_path = "C:\\Users\\xuefe\\Desktop\\大三下\\face\\人脸检测开发实践_比赛方案\\都挺好第1集.mp4"
image_path = "C:\\Users\\xuefe\\Desktop\\image\\"

times = 0
frameFrequency = 60

camera = cv2.VideoCapture(video_path)


while True:
    times += 1
    res, image = camera.read()
    if not res:
        print('not res , not image')
        break
    if ((times + 10) % 60) == 0 or ((times - 10) % 60) == 0:
        cv2.imwrite(image_path + str(times) + '.jpg', image)
        print(image_path + str(times) + '.jpg')

print('图片提取结束')
camera.release()
