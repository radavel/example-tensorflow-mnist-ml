from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784,10])) #weights
b = tf.Variable(tf.zeros([10])) #biases

#initializing the variables
sess.run(tf.global_variables_initializer())

y = tf.add(tf.matmul(x,W), b) # This means y = W + x + b


## Comming soon