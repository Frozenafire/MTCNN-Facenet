import tensorflow as tf
import facenet.align.detect_face as detect_face
from scipy import misc
import os

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


path = "C:\\Users\\xuefe\\Desktop\\facef\\a\\image\\"
img_dirs = os.listdir(path)
image_path = "C:\\Users\\xuefe\\Desktop\\facex\\b\\"

for im in img_dirs:
    img = path+'\\'+im
    img = misc.imread(img)
    bounding_boxes, face_points = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    if bounding_boxes.shape[0] != 1:
        continue
    # 通过人脸框置信度以及鼻子和眼睛的x坐标进行筛选
    if (bounding_boxes[0][4] <= 0.999) or (    # bounding_box[0][4]为人脸框的置信度
            (face_points[2][0] <= face_points[0][0] and face_points[2][0] <= face_points[1][0]) or (  # face_points表示
            face_points[2][0] >= face_points[0][0] and face_points[2][0] >= face_points[1][0])):  # 人脸的五个关键点
        continue                                  # 如果置信度过低或者人脸很歪则丢掉图片
    else:
        print(im, ':', bounding_boxes[0][4])
        misc.imsave(image_path+im, img)

