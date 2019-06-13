from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import os
import copy
import argparse
import facenet
import align.detect_face
import shutil
import cv2
import pandas as pd
import align.detect_face as detect_face


def sortface(path):
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

    def eyes(im):
        img = misc.imread(im)
        bounding_boxes, face_points = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if bounding_boxes.shape[0] == 0:
            return -1
        eye_dis = abs(face_points[0] - face_points[1])
        eye_mid = (face_points[0] + face_points[1])/2
        score1 = abs(face_points[2]-eye_mid)/eye_dis
        return round(score1[0], 2)

    def lap(im):
        img = cv2.imread(im)
        score2 = cv2.Laplacian(img, cv2.CV_64F).var()
        return score2//50

    face_score1 = []
    face_score2 = []
    face_dirs = os.listdir(path)
    face_dirs = [path+i for i in face_dirs]

    for face in face_dirs:
        print(face)
        s1 = eyes(face)
        face_score1.append(s1)
        if s1 >= 0:
            face_score2.append(lap(face))
        else:
            face_score2.append(0)

    data = {'a': face_dirs, 'b': face_score1, 'c': face_score2}
    data = pd.DataFrame(data)
    data = data.sort_values(['b', 'c'], ascending=[True, False])
#    data = data.sort_values('c', ascending=False)
    dir = data['a'].values.tolist()
    return dir


def main():
    pic_names = sortface("C:/Users/xuefe/Desktop/facef/faces/")
    ans_names = sortface("C:/Users/xuefe/Desktop/facex/facesf/")
#    ans_names = []
#    ans_names.append(pic_names[0])

    # Load the model
    minsize = 3  # minimum size of face
    threshold = [0.65, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

        tmp_image_paths = copy.copy(image_paths)
        img_list = []
        for image in tmp_image_paths:
            img = misc.imread(os.path.expanduser(image), mode='RGB')
            img_size = np.asarray(img.shape)[0:2]
            bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            if len(bounding_boxes) < 1:
                image_paths.remove(image)
                print("can't detect face, remove ", image)
                continue
            det = np.squeeze(bounding_boxes[0, 0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
            aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            img_list.append(prewhitened)
        images = np.stack(img_list)
        return images
    with tf.Graph().as_default():
        with tf.Session() as sess:
#            facenet.load_model("D:/python/models/20190218-164145.pb")
            facenet.load_model("D:/python/models/20180408-102900/20180408-102900.pb")
            # Get input and output tensors
            for i1 in pic_names:
                mindist = 0
                tag = 0
                for j1 in ans_names:
                    argv = [i1, j1]
                    args = parse_arguments(argv)
                    images = load_and_align_data(args.image_files, args.image_size, args.margin,
                                                 args.gpu_memory_fraction)
                    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

                    # Run forward pass to calculate embeddings
                    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                    emb = sess.run(embeddings, feed_dict=feed_dict)

                    nrof_images = len(args.image_files)

                    print('Images:')
                    for i in range(nrof_images):
                        print('%1d: %s' % (i, args.image_files[i]))
                    print('')

                    # Print distance matrix
                    print('Distance matrix')
                    print('    ', end='')
                    for i in range(nrof_images):
                        print('    %1d     ' % i, end='')
                    print('')
                    for i in range(nrof_images):
                        print('%1d  ' % i, end='')
                        for j in range(nrof_images):
                            dist = np.sqrt(np.sum(np.square(np.subtract(emb[i, :], emb[j, :]))))
                            if mindist == 0:
                                mindist = dist

                            if dist != 0 and dist < mindist:
                                # shutil.copyfile(argv[1],"F:/datasets/lfw/allgood_unique")
                                mindist = dist
                                if mindist < 1.01:
                                    tag = 1
                            print('  %1.4f  ' % dist, end='')
                        print('')

                    if tag == 1:
                        break
                # 偏差值大于1.1说明不是同一个人
                if mindist > 1.01:
                    ans_names.append(i1)
    print(len(ans_names))
    for i in ans_names:
        dir = "C:/Users/xuefe/Desktop/facex/facesff/"
        shutil.copy(i, dir)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    # parser.add_argument('model', type=str,
    # help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('image_files', type=str, nargs='+', help='Images to compare')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main()
