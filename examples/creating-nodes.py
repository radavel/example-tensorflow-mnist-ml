""" Importing the tensorflow framework """
import tensorflow as tf



""" Creating constants """
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicit

""" Print constant without initialize the sess """
print (node1, node2)



""" Creating the sess object """
sess = tf.Session()

""" Printing the constacts after run the sess """
print (sess.run([node1, node2]))



""" Crating a sum between node1 and node2 """
node3 = tf.add(node1, node2)

""" Printing the results of node1 + node2 without a sess """
print("node3: ", node3)

""" Printing the results of node3 = node1 + node2 with a sess """
print("sess.run(node3):", sess.run(node3))



""" Declaring the type of data """
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

""" Declaring a function for this model """
adder_node = a + b

""" Printing the results of the above function """
print(sess.run(adder_node, { a:59, b:852 }))
print(sess.run(adder_node, {a:[4,5], b:[584,1256]}))

