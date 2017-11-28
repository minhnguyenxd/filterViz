import dataset
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import utils
from math import sqrt
import cv2

# Adding Seed so that random initialization is consistent
from numpy.random import seed

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)

batch_size = 32

# Prepare input data
classes = ['dogs', 'cats']
num_classes = len(classes)

# 20% of the data will automatically be used for validation
validation_size = 0.2
img_size = 128
num_channels = 3
train_path = 'training_data'
PLOT_DIR = './out/plots_dog_cat'

# We shall load all the training and validation images and labels into memory using openCV and use that during training
data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)

# print("Complete reading input data. Will Now print a snippet of it")
# print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
# print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))


x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')

## labels
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

##Network graph params
filter_size_conv1 = 3
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64

fc_layer_size = 128


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

def create_convolutional_layer(input,
                               num_input_channels,
                               conv_filter_size,
                               num_filters):
    ## We shall define the weights that will be trained using create_weights function.
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    ## We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)

    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases

    ## We shall be using max-pooling.
    layer = tf.nn.max_pool(value=layer,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')
    ## Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)

    return layer


def create_flatten_layer(layer):
    # We know that the shape of the layer will be [batch_size img_size img_size num_channels]
    # But let's get it from the previous layer.
    layer_shape = layer.get_shape()

    ## Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
    num_features = layer_shape[1:4].num_elements()

    ## Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input,
                    num_inputs,
                    num_outputs,
                    use_relu=True):
    # Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer








#
# layer_conv1 = create_convolutional_layer(input=x,
#                                          num_input_channels=num_channels,
#                                          conv_filter_size=filter_size_conv1,
#                                          num_filters=num_filters_conv1)






# CONVOLUTION 1 - 1
with tf.variable_scope('conv1_1'):

    shape1 = [filter_size_conv1, filter_size_conv1, num_channels, num_filters_conv1]

    filter1_1 = tf.Variable(tf.truncated_normal(shape1, dtype=tf.float32,
                            stddev=1e-1), name='weights1_1')

    # ADD Weights to collection
    tf.add_to_collection('conv_weights', filter1_1)
    stride = [1,1,1,1]
    conv = tf.nn.conv2d(x, filter1_1, stride, padding='SAME', name='filer1')


    biases1 = tf.Variable(tf.constant(0.05, shape=[num_filters_conv1], dtype=tf.float32),
                          name='biases1_1')
    out = tf.nn.bias_add(conv, biases1)
    conv1_1 = tf.nn.relu(out)

# add output to collection
    tf.add_to_collection('conv_output', conv1_1)


# POOL 1
with tf.name_scope('pool1'):
    pool1_1 = tf.nn.max_pool(conv1_1,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='SAME',
                             name='pool1_1')
    # pool1_1_drop = tf.nn.dropout(pool1_1, keepRate1)







# layer_conv2 = create_convolutional_layer(input=layer_conv1,
#                                          num_input_channels=num_filters_conv1,
#                                          conv_filter_size=filter_size_conv2,
#                                          num_filters=num_filters_conv2)



# CONVOLUTION 1 - 2
with tf.name_scope('conv1_2'):

    shape2 = [filter_size_conv2, filter_size_conv2, num_filters_conv1, num_filters_conv2]

    filter1_2 = tf.Variable(tf.truncated_normal(shape2, dtype=tf.float32,
                            stddev=1e-1), name='weights1_2')
    # ADD Weights to collection
    tf.add_to_collection('conv_weights', filter1_2)
    stride = [1,1,1,1]
    conv2 = tf.nn.conv2d(pool1_1, filter1_2, stride, padding='SAME')
    biases2 = tf.Variable(tf.constant(0.05, shape=[num_filters_conv2], dtype=tf.float32),
                         trainable=True, name='biases1_2')
    out2 = tf.nn.bias_add(conv2, biases2)
    conv1_2 = tf.nn.relu(out2)
# add output to collection
    tf.add_to_collection('conv_output', conv1_2)


# POOL 2
with tf.name_scope('pool2'):
    pool1_2 = tf.nn.max_pool(conv1_2,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='SAME',
                             name='pool1_2')
    # pool1_1_drop = tf.nn.dropout(pool1_1, keepRate1)


#
# layer_conv3 = create_convolutional_layer(input=layer_conv2,
#                                          num_input_channels=num_filters_conv2,
#                                          conv_filter_size=filter_size_conv3,
#                                          num_filters=num_filters_conv3)


# CONVOLUTION 1 - 3
with tf.name_scope('conv1_3'):

    shape3 = [filter_size_conv3, filter_size_conv3, num_filters_conv2, num_filters_conv3]

    filter1_3 = tf.Variable(tf.truncated_normal(shape3, dtype=tf.float32,
                            stddev=1e-1), name='weights1_3')
    tf.add_to_collection('conv_weights', filter1_3)
    stride = [1,1,1,1]
    conv3 = tf.nn.conv2d(pool1_2, filter1_3, stride, padding='SAME')
    biases3 = tf.Variable(tf.constant(0.05, shape=[num_filters_conv3], dtype=tf.float32),
                         trainable=True, name='biases1_3')
    out3 = tf.nn.bias_add(conv3, biases3)
    conv1_3 = tf.nn.relu(out3)
    tf.add_to_collection('conv_output', conv1_3)


# POOL 3
with tf.name_scope('pool3'):
    pool1_3 = tf.nn.max_pool(conv1_3,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='SAME',
                             name='pool1_3')
    # pool1_1_drop = tf.nn.dropout(pool1_1, keepRate1)



layer_flat = create_flatten_layer(pool1_3)





#
# layer_fc1 = create_fc_layer(input=layer_flat,
#                             num_inputs=layer_flat.get_shape()[1:4].num_elements(),
#                             num_outputs=fc_layer_size,
#                             use_relu=True)
#
# layer_fc2 = create_fc_layer(input=layer_fc1,
#                             num_inputs=fc_layer_size,
#                             num_outputs=num_classes,
#                             use_relu=False)





#FULLY CONNECTED 1
with tf.name_scope('fc1') as scope:
    # Let's define trainable weights and biases.
    num_inputs_fc1 = layer_flat.get_shape()[1:4].num_elements()
    num_outputs_fc1 = fc_layer_size
    weights = create_weights(shape=[num_inputs_fc1, num_outputs_fc1])
    biases = create_biases(num_outputs_fc1)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    fc1_layer = tf.matmul(layer_flat, weights) + biases

    fc1_layer_relu = tf.nn.relu(fc1_layer)


# FULLY CONNECTED 2
with tf.name_scope('fc2') as scope:
    # Let's define trainable weights and biases.
    num_inputs_fc2 = fc_layer_size
    num_outputs_fc2 = num_classes
    weights = create_weights(shape=[num_inputs_fc2, num_outputs_fc2])
    biases = create_biases(num_outputs_fc2)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    fc2_layer = tf.matmul(fc1_layer_relu, weights) + biases

    #y_pred = tf.nn.softmax(fc2_layer, name='y_pred')



y_pred = tf.nn.softmax(fc2_layer, name='y_pred')

y_pred_cls = tf.argmax(y_pred, axis=1)



cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc2_layer,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




session = tf.Session()
session.run(tf.global_variables_initializer())



writer = tf.summary.FileWriter("output", session.graph_def)

# filter_summary = tf.image_summary(filter)

def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))


total_iterations = 0

saver = tf.train.Saver()


def train(num_iteration):
    global total_iterations

    for i in range(total_iterations,
                   total_iterations + num_iteration):

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)

        feed_dict_tr = {x: x_batch,
                        y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch,
                         y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_tr)

        if i % int(data.train.num_examples / batch_size) == 0:
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples / batch_size))

            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            saver.save(session, './dogs-cats-model')


            # writer.add_summary(conv1_1, global_step=None)
            # writer.close()


    #
    # # total_iterations += num_iteration
    # conv_weights = session.run([tf.get_collection('conv_weights')])
    # for i, c in enumerate(conv_weights[0]):
    #     plot_conv_weights(c, 'conv{}'.format(i))
    #
    # # First, pass the path of the image
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # # image_path=sys.argv[1]
    # image_path = 'testing_data/dogs/dog.1123.jpg'
    # filename = dir_path + '/' + image_path
    # image_size = 128
    # num_channels = 3
    # images = []
    # # Reading the image using OpenCV
    # image = cv2.imread(filename)
    # # Resizing the image to our desired size and preprocessing will be done exactly as done during training
    # image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
    # images.append(image)
    # images = np.array(images, dtype=np.uint8)
    # images = images.astype('float32')
    # images = np.multiply(images, 1.0 / 255.0)
    # # The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    # x_batch = images.reshape(1, image_size, image_size, num_channels)
    #
    #
    #
    # # get output of all convolutional layers
    # # here we need to provide an input image
    # conv_out = session.run([tf.get_collection('conv_output')], feed_dict={x: x_batch})
    # for i, c in enumerate(conv_out[0]):
    #     plot_conv_output(c, 'conv{}'.format(i))

train(num_iteration=800)
























