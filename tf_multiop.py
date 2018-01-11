import tensorflow as tf

print "One session with more than one operators."

a = tf.placeholder(tf.int32)
b = tf.placeholder(tf.int32)
mul_op = tf.multiply(a, b)
pow_op = tf.pow(a, b)
add_op = tf.add(mul_op, pow_op)
cos_op = tf.cos(45.)
with tf.Session() as sess:
    feed_dict = {a: 2, b: 3}
    print feed_dict
    print "After combination of some operators we see: "
    print sess.run([mul_op, pow_op, add_op, cos_op],
                   feed_dict=feed_dict)
