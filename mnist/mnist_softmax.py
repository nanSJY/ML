# -*- coding:utf-8 -*_

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# training set
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# parameters
W = tf.Variable(tf.zeros(shape=[784, 10]))
b = tf.Variable(tf.zeros(shape=[10]))

# input and output
x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10])
y = tf.nn.softmax(tf.matmul(x, W) + b)

# cross entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train_step = optimizer.minimize(cross_entropy)

# training
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(500):
    x_batch, y_batch = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: x_batch, y_: y_batch})

# evaluation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.validation.images, y_: mnist.validation.labels}))
