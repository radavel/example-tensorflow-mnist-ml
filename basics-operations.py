""" Using tensorflow to make operations """

import tensorflow as tf

# Build a graph.
width = tf.constant(158.0)
height = tf.constant(35.58)
area = width * height

# Launch the graph in a session.
sess = tf.Session()

# Evaluate the tensor `c`.
print(sess.run(area))