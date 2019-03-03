import os
import numpy as np
import tensorflow as tf
import input_data
import AlexNet


N_CLASSES = 5
IMG_W = 227
IMG_H = 227
BATCH_SIZE = 16
CAPACITY = 20
MAX_STEP = 5001
learning_rate = 0.0001

train_dir = 'data'
logs_train_dir = 'save'

train, train_label, val, val_label = input_data.get_files(train_dir, 0.2)
train_batch, train_label_batch = input_data.get_batch(train, train_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
val_batch, val_label_batch = input_data.get_batch(val, val_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

train_logits = AlexNet.network(train_batch, BATCH_SIZE, N_CLASSES)
train_loss = AlexNet.losses(train_logits, train_label_batch)
train_op = AlexNet.trainning(train_loss, learning_rate)
train_acc = AlexNet.evaluation(train_logits, train_label_batch)

test_logits = AlexNet.network(val_batch, BATCH_SIZE, N_CLASSES)
test_loss = AlexNet.losses(test_logits, val_label_batch)
test_acc = AlexNet.evaluation(test_logits, val_label_batch)

sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

try:
    for step in np.arange(MAX_STEP):
        if coord.should_stop():
            break
        _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])
        #val_loss, val_acc = sess.run([test_loss, test_acc])

        if step % 10 == 0:
            print('Step %d, train_loss=%.3f, train_accuracy=%.3f' % (step, tra_loss, tra_acc))

        if step % 500 == 0:
            checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')

finally:
    coord.request_stop()



















