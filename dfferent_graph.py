import tensorflow as tf

# Define 2 graphs
g1 = tf.Graph()
g2 = tf.Graph()

# The "v" is a variable in these two graphs
with g1.as_default():
    tf.get_variable("v", [2, 3], initializer=tf.zeros_initializer())

with g2.as_default():
    tf.get_variable("v", [2, 1], initializer=tf.ones_initializer())

with tf.Session(graph=g1) as sess:
    sess.run(tf.global_variables_initializer())
    with tf.variable_scope("", reuse=tf.AUTO_REUSE):
        print "v in g1 = "
        print sess.run(tf.get_variable("v"))

with tf.Session(graph=g2) as sess:
    sess.run(tf.global_variables_initializer())
    with tf.variable_scope("", reuse=tf.AUTO_REUSE):
        print "v in g2 = "
        print sess.run(tf.get_variable("v"))
