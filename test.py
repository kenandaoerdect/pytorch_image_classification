from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import AlexNet


def evaluate_one_image(image_array):
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 5

        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 227, 227, 3])

        logit = AlexNet.network(image, BATCH_SIZE, N_CLASSES)
        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[227, 227, 3])

        logs_train_dir = 'save'
        saver = tf.train.Saver()
        with tf.Session() as sess:
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
            prediction = sess.run(logit, feed_dict={x: image_array})
            print(prediction)
            max_index = np.argmax(prediction)
            if max_index == 0:
                result = ('abyssinian possibility： %.6f' % prediction[:, 0])
            elif max_index == 1:
                result = ('american_bulldog possibility： %.6f' % prediction[:, 1])
            elif max_index == 2:
                result = ('Birman possibility： %.6f' % prediction[:, 2])
            elif max_index == 3:
                result = ('Sphynx possibility： %.6f' % prediction[:, 3])
            else:
                result = ('yorkshire_terrier possibility： %.6f' % prediction[:, 4])
            return result


if __name__ == '__main__':
    img = Image.open(r'D:\DeepLearning\animal_recognition\test\yorkshire_terrier\yorkshire_terrier_198.jpg')
    plt.imshow(img)
    imag = img.resize([227, 227])
    image = np.array(imag)
    print(evaluate_one_image(image))
    plt.show()