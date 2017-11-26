# coding=utf-8
import tensorflow as tf
import numpy as np

# Read this code-sheet you must have basic knowledge of artificial neural network(ANN).
# https://en.wikipedia.org/wiki/Artificial_neural_network

# I promise that this code-sheet is a "hello,world" of "hello,world" samples comparing
# with others.

# I will define a network with
#
# 1. One layer for input
# 2. One hide-layer with four nodes
# 3. One layer for output

# For demo, a lot of data and statistic will be cleared, and a scenario will be defined.

# 1. scenario: A factory which build components for some machines. We must use this ANN to find out
#              what component bad, what good.
#
# 2. component: Just a cube which has length, width, height: L, W, H
#
# 3. data & statistic:
#               3.1 The collection of all cubes will be declared as one-line vector, see [L, W, H]
#               3.2 For demo, [L, W, H] will be initialized with hard-coding randomly.
#               3.3 Weight: From one layer to another we need weight, see Wt.
#               3.4 Bias: For this simple demo, I don't want to bring this concept, it'll be used in
#                   the future samples.

# Define cube(component) collection.
# It'll look like:
# [[23, 53, 12], [24, 67, 24], [34, 68, 24],..... ]
# The random is being generated with the standard-deviation.
# Don't care about deviation, you can do all what you can to generate this array(matrix).

# As master of this factory I want to checkout fist N couple of components.
N = 1000

cubes = tf.constant(np.random.normal(loc=20.0, scale=20.0, size=(N, 3)))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    components = sess.run(cubes)
    for component in components:
        print component
