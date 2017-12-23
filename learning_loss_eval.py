import tensorflow as tf
import numpy as np

# Softmax
Y = tf.constant([[0.3, 0.4, 0.5]])  # estimated
Y_ = tf.constant([[0.3, 0.4, 0.5]])  # real

out1 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Y))
with tf.Session() as sess:
    print sess.run(out1)

# clip value
input_x = tf.constant([[0.3, -0.4, 0.5], [8, 2, 1], [-3, 6, 8]])
clip_x = tf.clip_by_value(input_x, 0, 6)
with tf.Session() as sess:
    print sess.run(clip_x)

# Softmax, reduce_mean with logits
print "\nSoftmax, reduce_mean with logits:\n"
print "y_"
y_ = [
    [0., 1.],
    [1., 0.],
    [1., 1.]
]
print y_
print "logits"
logits = [
    [3., 7.],
    [9., 4.],
    [8., 1.]
]
print logits
y = tf.nn.softmax(logits=logits)
cross_entropy = -tf.reduce_mean(y_ * tf.log(y))
with tf.Session() as sess:
    softmax = sess.run(y)
    print "softmax:"
    print softmax
    print "Do math...."
    print sess.run(cross_entropy)
    print "Do manually...."
    print (
        -(
            0 * np.log(softmax[0][0]) + 1 * np.log(softmax[0][1]) +
            1 * np.log(softmax[1][0]) + 0 * np.log(softmax[1][1]) +
            1 * np.log(softmax[2][0]) + 1 * np.log(softmax[2][1])
        ) / 3)
    print "builtIn softmax_cross_entropy_with_logits:", sess.run(
        tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits)))

# print "\ntry something again...."
# print "y_"
# y_ = [
#     [1.],
#     [1.],
#     [1.]
# ]
# print y_
# print "logits"
# logits = [
#     [3.],
#     [9.],
#     [8.]
# ]
# print logits
# y = tf.nn.softmax(logits=logits)
# cross_entropy = -tf.reduce_mean(y_ * tf.log(y))
# with tf.Session() as sess:
#     softmax = sess.run(y)
#     print "softmax:"
#     print softmax
#     print "Do math...."
#     print sess.run(cross_entropy)
#     print "Do manually...."
#     print (
#         -(
#             1 * np.log(softmax[0][0]) +
#             1 * np.log(softmax[1][0]) +
#             1 * np.log(softmax[2][0])
#         ) / 3)
#     print "builtIn softmax_cross_entropy_with_logits:", sess.run(
#         tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits)))
