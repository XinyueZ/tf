# coding=utf-8
import numpy as np
import tensorflow as tf

# Read this code-sheet you must have basic knowledge of artificial neural network(ANN).
# https://en.wikipedia.org/wiki/Artificial_neural_network

# I promise that this code-sheet is a "hello,world" of "hello,world" samples comparing
# with others.

# In this example, there's no loss-function, optimizer, it is a filter logical with matrix functions
# of TF. You can do this with python numpy of cos, however, we are introducing the Tensorflow.


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
#               3.2 For demo, [L, W, H] will be initialized with hard-coding randomly, see cubes
#               3.3 Weight: From one layer to another we need weight, see Weights.
#               3.4 Bias: For this simple demo, I don't want to bring this concept, it'll be used in
#                   the future samples.

# Define cube(component) collection.
# It'll look like:
# [[23, 53, 12], [24, 67, 24], [34, 68, 24],..... ]
# The random is being generated with the standard-deviation.
# Don't care about deviation, you can do all what you can to generate this array(matrix).

# The whole process will be like this(one component):

#         Input                                           weight-0                                        step 1 finished                                  weight-1                     Done
#
#
# [                   ]                          [                   ]                             [                   ]                            [                   ]        [                   ]
#
#                                                     34, 45 , 5                                                                                             77
#
#
#     [34, 54, 57]        Do Multiplication           35, 55 , 5        get layer 0 > layer 1          [  55, 56 , 66 ]      Do Multiplication               67              =>          -46
#
#
#
#                                                     55, 56 , 5                                                                                             56
#
#
# [                   ]                          [                   ]                              [                   ]                           [                   ]        [                   ]




# As master of this factory I want to checkout fist N couple of components.
N = 100

cubes = tf.constant(np.random.normal(loc=0.0, scale=1.0, size=(N, 3)))
x = tf.placeholder(tf.float32, (N, 3))

# Weight for all steps, see below for details.
Weights_1 = tf.constant(np.random.normal(loc=0.0, scale=1.0, size=(3, 3)), dtype=tf.float32)
Weights_2 = tf.constant(np.random.normal(loc=0.0, scale=1.0, size=(3, 1)), dtype=tf.float32)

# Make an artificial neural network(ANN) with
# layer 0:   input layer
# layer 1:   hide layer
# layer 2:   output layer
# Rule:
#       when output is > 0 ->>> it is a good component(cube).
#       otherwise ->>> it is out of quality.
# Transfer between layers

step_1 = tf.matmul(x, Weights_1)  # input -> layer 1
y = tf.matmul(step_1, Weights_2)  # output

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    components = sess.run(cubes)
    outputs = sess.run(y, feed_dict={x: components})
    index = 0
    for output in outputs:
        component = components[index]
        if output[0] > 0:
            print "\t\t\t✓\t\t\t", \
                "X= Length:", '%2.4f' % component[0], \
                "Width:", '%2.4f' % component[1], \
                "Height:", '%2.4f' % component[2], \
                'Y= %2.4f' % output[0]
        else:
            print "\t\t\t✗\t\t\t", \
                "X= Length:", '%2.4f' % component[0], \
                "Width:", '%2.4f' % component[1], \
                "Height:", '%2.4f' % component[2], \
                'Y= %2.4f' % output[0]
        index = index + 1
