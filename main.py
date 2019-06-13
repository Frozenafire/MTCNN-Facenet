import tensorflow as tf
import facenet.align.detect_face as detect_face
import cv2
from scipy import misc


minsize = 3  # 脸部最小的大小
threshold = [0.65, 0.7, 0.7]  # 三个步骤的阈值
factor = 0.709  # 用于在图像中检测的人脸大小的缩放金字塔的因子

# 分配给 tensorflow 的 gpu 的显存大小: GPU 实际显存 * 0.7
gpu_memory_fraction = 0.7

# 创建 tensorflow 网络并加载参数
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

# image_path = "C:\\Users\\xuefe\\Desktop\\1.jpg"
image_path = "C:\\Users\\xuefe\\Desktop\\new\\b\\8800_4.png"
# image_path = "C:\\Users\\xuefe\\Desktop\\faces\\396.jpg"
img = misc.imread(image_path)  # 使用 opencv 的方法读取图片
print(img.ndim)
print(img.shape)
print(img[:,:,0:3].shape)

bounding_boxes, face_points = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

nrof_faces = bounding_boxes.shape[0]  # 人脸数目
print('  :', bounding_boxes[0][4])
print('找到的人脸数目为：{}'.format(nrof_faces))
print("打印 bounding_box 的内容:\n", bounding_boxes)
print("打印 face_points 的内容：\n", face_points)

# 画人脸框
for index, face_position in enumerate(bounding_boxes):
    face_position = face_position.astype(int)
    cv2.rectangle(img, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 2)
    # 人脸特征点
    for i in range(5):
        cv2.circle(img, (face_points[i, index], face_points[i + 5, index]), 3, (255, 255, 0), -1)
cv2.imshow("img", img)
# cv2.imwrite("C:\\Users\\xuefe\\Desktop\\8800-1.jpg",img)
cv2.waitKey()
