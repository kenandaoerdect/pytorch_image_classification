import tensorflow as tf


def network(images, batch_size, n_classes):

    weights = tf.Variable(tf.truncated_normal(shape=[11, 11, 3, 96], dtype=tf.float32, stddev=1e-1), dtype=tf.float32)
    biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[96]), dtype=tf.float32)
    conv = tf.nn.conv2d(images, weights, strides=[1, 4, 4, 1], padding='SAME')
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    weights = tf.Variable(tf.truncated_normal(shape=[5, 5, 96, 256], dtype=tf.float32, stddev=1e-1), dtype=tf.float32)
    biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[256]), dtype=tf.float32)
    conv = tf.nn.conv2d(pool1, weights, strides=[1, 1, 1, 1], padding='SAME')
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 384], dtype=tf.float32, stddev=1e-1), dtype=tf.float32)
    biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[384]), dtype=tf.float32)
    conv = tf.nn.conv2d(pool2, weights, strides=[1, 1, 1, 1], padding='SAME')
    pre_activation = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(pre_activation)

    weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 384, 384], dtype=tf.float32, stddev=1e-1), dtype=tf.float32)
    biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[384]), dtype=tf.float32)
    conv = tf.nn.conv2d(conv3, weights, strides=[1, 1, 1, 1], padding='SAME')
    pre_activation = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(pre_activation)

    weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 384, 256], dtype=tf.float32, stddev=1e-1), dtype=tf.float32)
    biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[256]), dtype=tf.float32)
    conv = tf.nn.conv2d(conv4, weights, strides=[1, 1, 1, 1], padding='SAME')
    pre_activation = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu(pre_activation)
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    reshape = tf.reshape(pool5, shape=[batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = tf.Variable(tf.truncated_normal(shape=[dim, 4096], stddev=0.005, dtype=tf.float32), dtype=tf.float32)
    biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[4096]))
    FC = tf.nn.relu(tf.matmul(reshape, weights) + biases)
    FC1 = tf.nn.dropout(FC, keep_prob=0.5)

    weights = tf.Variable(tf.truncated_normal(shape=[4096, 2048], stddev=0.005, dtype=tf.float32), dtype=tf.float32)
    biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[2048]), dtype=tf.float32)
    FC = tf.nn.relu(tf.matmul(FC1, weights) + biases)
    FC2 = tf.nn.dropout(FC, keep_prob=0.5)

    weights = tf.Variable(tf.truncated_normal(shape=[2048, n_classes], stddev=0.005, dtype=tf.float32), dtype=tf.float32)
    biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[n_classes]), dtype=tf.float32)
    softmax_linear = tf.add(tf.matmul(FC2, weights), biases)

    return softmax_linear


def losses(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(cross_entropy)
    return loss


def trainning(loss, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    global_step = tf.Variable(0, trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    correct = tf.cast(correct, tf.float16)
    accuracy = tf.reduce_mean(correct)
    return accuracy








