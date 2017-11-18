import tensorflow as tf

print "Do multiplication in different way"

print "1. Use normal math-op"
a = tf.constant(2)
b = tf.constant(3)
with tf.Session() as sess:
    print "a:%i" % sess.run(a), "b:%i" % sess.run(b)
    print "Multiplication with constants: %i" % sess.run(a * b)

print "2. Use multiplication-op"
a = tf.placeholder(tf.int8)
b = tf.placeholder(tf.int8)
mul = tf.multiply(a, b)  # Multiplication graph
with tf.Session() as sess:
    feed_dict = {a: 2, b: 3}
    print feed_dict
    print "Multiplication with constants: %i" % sess.run(mul, feed_dict=feed_dict)

print "3. Matrix multiplication"
matrix_1 = tf.constant([[3., 3.]])
matrix_2 = tf.constant([[2.], [2.]])
product = tf.matmul(matrix_1, matrix_2)
with tf.Session() as sess:
    res = sess.run(product)
    print res

