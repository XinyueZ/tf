from numpy.random import RandomState
import tensorflow as tf
import numpy as np

rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

print X
print Y

print X[0:3]
print Y[0:3]

y_ = [
    [0, 1],
    [1, 0],
    [1, 1]
]
y = [
    [0.5, 0.3],
    [0.3, 0.45],
    [0.4, 0.25]
]
cross_entropy = -tf.reduce_mean(y_ * tf.log(y))
with tf.Session() as sess:
    print sess.run(cross_entropy)

print (
    -(
        0 * np.log(0.5) + 1 * np.log(0.3) +
        1 * np.log(0.3) + 0 * np.log(0.45) +
        1 * np.log(0.4) + 1 * np.log(0.25)
    ) / 6)

y_ = [
    [1],
    [1],
    [1]
]
y = [
    [0.5],
    [0.45],
    [0.25]
]
cross_entropy = -tf.reduce_mean(y_ * tf.log(y))
with tf.Session() as sess:
    print sess.run(cross_entropy)

print (
    -(
        1 * np.log(0.5) +
        1 * np.log(0.45) +
        1 * np.log(0.25)
    ) / 3)
