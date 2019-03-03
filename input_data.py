import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


train_dir = 'data'

abyssinian = []
label_abyssinian = []
american_bulldog = []
label_american_bulldog = []
Birman = []
label_Birman = []
Sphynx = []
label_Sphynx = []
yorkshire_terrier = []
label_yorkshire_terrier = []

def get_files(file_dir, ratio):
    for file in os.listdir(file_dir + '/abyssinian'):
        abyssinian.append(file_dir + '/abyssinian' + '/' + file)
        label_abyssinian.append(0)
    for file in os.listdir(file_dir + '/american_bulldog'):
        american_bulldog.append(file_dir + '/american_bulldog' + '/' + file)
        label_american_bulldog.append(1)
    for file in os.listdir(file_dir + '/Birman'):
        Birman.append(file_dir + '/Birman' + '/' + file)
        label_Birman.append(2)
    for file in os.listdir(file_dir + '/Sphynx'):
        Sphynx.append(file_dir + '/Sphynx' + '/' + file)
        label_Sphynx.append(3)
    for file in os.listdir(file_dir + '/yorkshire_terrier'):
        yorkshire_terrier.append(file_dir + '/yorkshire_terrier' + '/' + file)
        label_yorkshire_terrier.append(4)

    image_list = np.hstack((abyssinian, american_bulldog, Birman, Sphynx, yorkshire_terrier))
    label_list = np.hstack((label_abyssinian, label_american_bulldog, label_Birman, label_Sphynx, label_yorkshire_terrier))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    all_image_list = list(temp[:, 0])
    all_label_list = list(temp[:, 1])

    n_sample = len(all_label_list)
    n_val = int(math.ceil(n_sample*ratio))
    n_train = n_sample-n_val

    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]
    return tra_images, tra_labels, val_images, val_labels


def get_batch(image, label, image_W, image_H, batch_size, capacity):
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])

    image = tf.image.decode_jpeg(image_contents, channels=3)

    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=32,
                                              capacity=capacity)
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch
