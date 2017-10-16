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
print(sess.run(adder_node, { a:3, b:4.5 }))
print(sess.run(adder_node, {a:[1,3], b:[2,4]}))


add_and_triple = adder_node * 3
print(sess.run(add_and_triple, {a:3, b:4.5}))


W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)

linear_model = W * x + b

init = tf.global_variables_initializer()

sess.run(init)

print(sess.run(linear_model, {x: [1, 2, 3, 4]}))