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

# The whole process will be like this:

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

def format_num(num):
    return "{:.20f}".format(num)


# As master of this factory I want to checkout fist N couple of components.
N = 100

cubes = tf.constant(np.random.normal(loc=10.0, scale=30.0, size=(N, 3)))

# Weight for all steps, see below for details.
Weights_1 = tf.constant(np.random.normal(loc=1.0, scale=5.0, size=(3, 3)))
Weights_2 = tf.constant(np.random.normal(loc=1.0, scale=5.0, size=(3, 1)))

# Make an artificial neural network(ANN) with
# layer 0:   input layer
# layer 1:   hide layer
# layer 2:   output layer
# Rule:
#       when output is > 0 ->>> it is a good component(cube).
#       otherwise ->>> it is out of quality.
# Transfer between layers

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    components = sess.run(cubes)
    print "################################################"
    print "Component(cube) list:"
    print "################################################"
    for component in components:
        reshape_component = tf.reshape(component, [1, 3])
        reshaped = sess.run(reshape_component)
        step_1 = tf.matmul(reshaped, Weights_1)  # input -> layer 1
        step_2 = tf.matmul(step_1, Weights_2)  # output
        sess.run(step_1)
        output = sess.run(step_2)
        if output[0][0] > 0:
            print "Component ====> Length = ", format_num(
                reshaped[0][0]), "\t", ", Width = ", format_num(
                reshaped[0][1]), "\t", ", Height = ", "\t", format_num(
                reshaped[0][2]), "\t", "====>", "\t", format_num(
                output[0][0]), "====>\t\t✓"
        else:
            print "Component ====> Length = ", format_num(
                reshaped[0][0]), "\t", ", Width = ", format_num(
                reshaped[0][1]), "\t", ", Height = ", "\t", format_num(
                reshaped[0][2]), "\t", "====>", "\t", format_num(
                output[0][0]), "====>\t\t✗"
