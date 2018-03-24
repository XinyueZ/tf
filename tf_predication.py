import random

import numpy as np
import tensorflow as tf

from data_predication import features


def create_feature_sets_and_labels(features, test_size=0.3):
    # shuffle out features and turn into np.array
    random.shuffle(features)
    features = np.array(features)

    # split a portion of the features into tests
    testing_size = int(test_size * len(features))

    # create train and test lists
    train_x = list(features[:, 0][:-testing_size])
    train_y = list(features[:, 1][:-testing_size])
    test_x = list(features[:, 0][-testing_size:])
    test_y = list(features[:, 1][-testing_size:])

    return train_x, train_y, test_x, test_y


train_x, train_y, test_x, test_y = create_feature_sets_and_labels(features)

# hidden layers and their nodes
n_nodes_hl1 = 20
n_nodes_hl2 = 20

# classes in our output
n_classes = 2
# iterations and batch-size to build out model
learning_times = 3000
learning_rate = 1
batch_size = 4

inputs = tf.placeholder(tf.float32)
outputs = tf.placeholder(tf.float32)

# random weights and bias for our layers
hidden_1_layer = {'f_fum': n_nodes_hl1,
                  'weight': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum': n_nodes_hl2,
                  'weight': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))}

output_layer = {'f_fum': None,
                'weight': tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                'bias': tf.Variable(tf.random_normal([n_classes])), }


# our predictive model's definition
def neural_network_model(input_data):
    # hidden layer 1: (data * W) + b
    l1 = tf.add(tf.matmul(input_data, hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.sigmoid(l1)

    # hidden layer 2: (hidden_layer_1 * W) + b
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.sigmoid(l2)

    # output: (hidden_layer_2 * W) + b
    output_data = tf.matmul(l2, output_layer['weight']) + output_layer['bias']

    return output_data


# training our model
def train_neural_network():
    # use the model definition
    prediction = neural_network_model(inputs)

    # formula for cost (error)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=outputs))

    # optimize for cost using GradientDescent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Tensorflow session
    with tf.Session() as sess:
        tf.summary.FileWriter('log_ANN_graph', sess.graph)
        # initialize our variables
        sess.run(tf.global_variables_initializer())

        # loop through specified number of iterations
        for epoch in range(learning_times):
            i = 0
            # handle batch sized chunks of training data
            for _ in train_x:
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, loss], feed_dict={inputs: batch_x, outputs: batch_y})
                i += batch_size
                last_cost = c
                if i >= len(train_x): break

            # print cost updates along the way
            if (epoch % (learning_times / 5)) == 0:
                print('Epoch', epoch, 'completed out of', learning_times, 'cost:', last_cost)

        # print accuracy of our model
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(outputs, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({inputs: test_x, outputs: test_y}))

        output_weight = sess.run(output_layer['weight'])
        output_bias = sess.run(output_layer['bias'])

        # print predictions using our model
        for i, t in enumerate(test_x):
            print ('prediction for:', test_x[i])
            output = prediction.eval(feed_dict={inputs: [test_x[i]]})
            # normalize the prediction values
            print(tf.sigmoid(output[0][0]).eval(), tf.sigmoid(output[0][1]).eval())

        return output_weight, output_bias


output_weight, output_bias = train_neural_network()
print("final output_weight:\n{}".format(output_weight))
print("final output_bias: {}".format(batch_size))
