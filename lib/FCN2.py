from __future__ import print_function

import datetime
import os

import TensorflowUtils as utils
import numpy as np
import read_Musicdataset as music_data
import tensorflow as tf
from six.moves import xrange
import cPickle as pickle
import sys
sys.path.insert(0, '.')
import BatchDatasetReader as dataset
import utils as util
import os.path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "1", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
tf.flags.DEFINE_string("music_data_dir", "../Musicdevkit/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

MAX_ITERATION = int(5e6 + 1)
NUM_OF_CLASSESS = 65
IMAGE_SIZE = 224 * 3

def save_object(obj, filename):
    # Save whole object (used for dataset-readers)
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    # load whole object (used for dataset-readers)
    with open(filename, 'rb') as input:
        return pickle.load(input)

def inference(image, keep_prob_conv, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("Building the network ...")

    with tf.variable_scope("inference"):

        # build here the neural network; later it needs to be put in a class for better organization
        # build the first layer; input_size = 448, output_size = 224
        W1 = utils.weight_variable([3, 3, 1, 32], name="W1")
        b1 = utils.bias_variable([32], name="b1")
        conv1 = utils.conv2d_basic(image, W1, b1, name="conv1")
        relu1 = tf.nn.relu(conv1, name="relu1")
        pool1 = utils.max_pool_2x2(relu1)
        dropout1 = tf.nn.dropout(pool1, keep_prob=keep_prob_conv)

        # build the second layer; input_size = 224, output_size = 112
        W2 = utils.weight_variable([3, 3, 1, 64], name="W2")
        b2 = utils.bias_variable([64], name="b2")
        conv2 = utils.conv2d_basic(image, W2, b2, name="conv2")
        relu2 = tf.nn.relu(conv2, name="relu2")
        pool2 = utils.max_pool_2x2(relu2)
        dropout2 = tf.nn.dropout(pool2, keep_prob=keep_prob_conv)

        # build the third layer; input_size = 112, output_size = 56
        W3 = utils.weight_variable([3, 3, 64, 128], name="W3")
        b3 = utils.bias_variable([128], name="b3")
        conv3 = utils.conv2d_basic(dropout2, W3, b3, name="conv3")
        relu3 = tf.nn.relu(conv3, name="relu3")
        pool3 = utils.max_pool_2x2(relu3)
        dropout3 = tf.nn.dropout(pool3, keep_prob=keep_prob_conv)

        # build the fourth layer; input_size = 56, output_size = 28
        W4 = utils.weight_variable([3, 3, 128, 256], name="W4")
        b4 = utils.bias_variable([256], name="b4")
        conv4 = utils.conv2d_basic(dropout3, W4, b4, name="conv4")
        relu4 = tf.nn.relu(conv4, name="relu4")
        pool4 = utils.max_pool_2x2(relu4)
        dropout4 = tf.nn.dropout(pool4, keep_prob=keep_prob_conv)

        # build the fifth layer; input_size = 28, output_size = 14
        W5 = utils.weight_variable([3, 3, 256, 512], name="W5")
        b5 = utils.bias_variable([512], name="b5")
        conv5 = utils.conv2d_basic(dropout4, W5, b5, name="conv5")
        relu5 = tf.nn.relu(conv5, name="relu5")
        pool5 = utils.max_pool_2x2(relu5)
        dropout5 = tf.nn.dropout(pool5, keep_prob=keep_prob_conv)

        # build the sixth layer; input_size = 14, output_size = 7
        W6 = utils.weight_variable([3, 3, 512, 512], name="W6")
        b6 = utils.bias_variable([512], name="b6")
        conv6 = utils.conv2d_basic(dropout5, W6, b6, name="conv6")
        relu6 = tf.nn.relu(conv6, name="relu6")
        pool6 = utils.max_pool_2x2(relu6)
        dropout6 = tf.nn.dropout(pool6, keep_prob=keep_prob_conv)

        # build the seventh layer, input_size = 7, output_size = 7
        W7 = utils.weight_variable([3, 3, 512, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(dropout6, W7, b7, name="conv7")
        relu7 = tf.nn.relu(conv7, name="relu8")
        dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        # build the eighth layer, input size = 7, output_size = 7
        W8 = utils.weight_variable([3, 3, 4096, 4096], name="W8")
        b8 = utils.bias_variable([4096], name="b8")
        conv8 = utils.conv2d_basic(dropout7, W8, b8, name="conv8")
        relu8 = tf.nn.relu(conv8, name="relu8")
        dropout8 = tf.nn.dropout(relu8, keep_prob=keep_prob)

        # build the final (tenth) convolutional layer; input_size = 7, output_size = 7
        W9 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W9")
        b9 = utils.bias_variable([NUM_OF_CLASSESS], name="b9")
        conv9 = utils.conv2d_basic(dropout8, W9, b9, name="conv9")


        conv_t3_labels = upsample(pool5, pool4,pool3,pool2,conv9, image, "upscale_labels", NUM_OF_CLASSESS)
        annotation_pred = tf.argmax(conv_t3_labels, dimension=3, name="prediction")

        conv_t3_objects = upsample(pool5, pool4,pool3,pool2,conv9, image, "upscale_objects", 1)

    return tf.expand_dims(annotation_pred, dim=3), conv_t3_labels, conv_t3_objects

def upsample(pool5, pool4,pool3,pool2, conv9, image, scope, output_class):
    with tf.variable_scope(scope):
        # do the upscaling using 2 fuse layers
        deconv_shape1 = pool5.get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv9, W_t1, b_t1, output_shape=tf.shape(pool5))
        fuse_1 = tf.add(conv_t1, pool5, name="fuse_1")

        deconv_shape2 = pool4.get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(pool4))
        fuse_2 = tf.add(conv_t2, pool4, name="fuse_2")

        deconv_shape3 = pool3.get_shape()
        W_t3 = utils.weight_variable([4, 4, deconv_shape3[3].value, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([deconv_shape3[3].value], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=tf.shape(pool3))
        fuse_3 = tf.add(conv_t3, pool3, name="fuse_3")

        deconv_shape4 = pool2.get_shape()
        W_t4 = utils.weight_variable([4, 4, deconv_shape4[3].value, deconv_shape3[3].value], name="W_t4")
        b_t4 = utils.bias_variable([deconv_shape4[3].value], name="b_t4")
        conv_t4 = utils.conv2d_transpose_strided(fuse_3, W_t4, b_t4, output_shape=tf.shape(pool2))
        fuse_4 = tf.add(conv_t4, pool2, name="fuse_4")

        # do the final upscaling
        shape = tf.shape(image)
        deconv_shape5 = tf.stack([shape[0], shape[1], shape[2], output_class])
        W_t5 = utils.weight_variable([16, 16, output_class, deconv_shape4[3].value], name="W_t5")
        b_t5 = utils.bias_variable([output_class], name="b_t5")
        conv_t5 = utils.conv2d_transpose_strided(fuse_4, W_t5, b_t5, output_shape=deconv_shape5, stride =2)
        return conv_t5

def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    keep_probability_conv = tf.placeholder(tf.float32, name="keep_probability_conv")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="input_image")
    annotation_labels = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation_labels")
    annotation_objects = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation_objects")

    pred_annotation, logits, pred_objects = inference(image, keep_probability_conv, keep_probability)
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth_labels", tf.cast(annotation_labels, tf.uint8), max_outputs=2)
    tf.summary.image("ground_truth_objects", tf.cast(annotation_objects, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_objectness", tf.cast(pred_objects, tf.uint8), max_outputs=2)

    loss_labels = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation_labels, squeeze_dims=[3]),
                                                                          name="entropy")))

    loss_objects = tf.scalar_mul(0.006, tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(pred_objects, tf.cast(annotation_objects,tf.float32))), reduction_indices=(1,2)))))

    #loss = tf.add(loss_labels, loss_objects)
    loss = loss_labels

    tf.summary.scalar("entropy", loss)

    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    if False:
        # try to load cached Data
        fname_train = "train_dsreader" + str(IMAGE_SIZE)
        fname_valid = "valid_dsreader" + str(IMAGE_SIZE)
        if os.path.isfile(fname_train) and os.path.isfile(fname_valid):
            # load cached
            train_dataset_reader = load_object(fname_train)
            validation_dataset_reader = load_object(fname_valid)
        else:
            # Load Data from disk and cache it
            print("Setting up image reader...")
            train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
            train_records, valid_records = music_data.read_dataset(FLAGS.data_dir)
            print(len(train_records))
            print(len(valid_records))

            print("Setting up dataset reader")
            image_options = {'resize': False, 'resize_size': IMAGE_SIZE}
            if FLAGS.mode == 'train':
                train_dataset_reader = dataset.BatchDatset(train_records, image_options)
            validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)

            if train_dataset_reader is not None:
                save_object(train_dataset_reader, fname_train)
            save_object(validation_dataset_reader, fname_valid)
    else:
        # just load from disk
        print("Setting up image reader...")
        #train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
        train_records, valid_records = music_data.read_dataset(FLAGS.music_data_dir)
        print(len(train_records))
        print(len(valid_records))

        print("Setting up dataset reader")
        image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
        if FLAGS.mode == 'train':
            train_dataset_reader = dataset.BatchDatset(train_records, image_options)
        validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)

    sess = tf.Session()


    print("Setting up Saver...")
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)


    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1]) # get the step from the last checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    else:
        step = 0

    if FLAGS.mode == "train":
        for itr in xrange(step, MAX_ITERATION):
            train_images, train_m_annotations, train_o_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            feed_dict = {image: train_images, annotation_labels: train_m_annotations,annotation_objects: train_o_annotations, keep_probability_conv: 0.85, keep_probability: 0.85}
            sess.run(train_op, feed_dict=feed_dict)

            if itr % 10 == 0:
                train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                # train_loss = sess.run(loss_labels, feed_dict=feed_dict)
                # print("Step: %d, labels_loss:%g" % (itr, train_loss))
                # train_loss = sess.run(loss_objects, feed_dict=feed_dict)
                # print("Step: %d, objects_loss:%g" % (itr, train_loss))
                summary_writer.add_summary(summary_str, itr)

            if itr % 500 == 0 and itr != 0:
                valid_images, valid_m_annotations, valid_o_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
                valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation_labels: valid_m_annotations,annotation_objects: valid_o_annotations, keep_probability_conv: 1.0,
                                                       keep_probability: 1.0})
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

    elif FLAGS.mode == "visualize":
        number_of_batches = 100 # number of batches you want to visualize (workaround to not get out of memory errors)
        for i in xrange(number_of_batches):
            valid_images, valid_m_annotations, valid_o_annotations = validation_dataset_reader.next_batch(
                FLAGS.batch_size)
            pred_a, pred_o = sess.run([pred_annotation,pred_objects], feed_dict={image: valid_images, annotation_labels: valid_m_annotations, annotation_objects: valid_o_annotations,
                                                        keep_probability_conv: 1.0, keep_probability: 1.0})

            valid_o_annotations = np.squeeze(valid_o_annotations, axis=3)
            valid_m_annotations = np.squeeze(valid_m_annotations, axis=3)
            pred_a = np.squeeze(pred_a, axis=3)
            pred_o = np.squeeze(pred_o, axis=3)

            for itr in range(FLAGS.batch_size):
                utils.save_image(valid_m_annotations[itr].astype(np.uint8), "../"+FLAGS.logs_dir + "trained_images/ground_truth", name="gt_" + str(i*FLAGS.batch_size+itr)+ "_m")
                utils.save_image(valid_o_annotations[itr].astype(np.uint8), "../"+FLAGS.logs_dir + "trained_images/ground_truth", name="gt_" + str(i*FLAGS.batch_size+itr)+ "_o")
                utils.save_image(pred_a[itr].astype(np.uint8), "../"+FLAGS.logs_dir + "trained_images/prediction", name="pred_" + str(i*FLAGS.batch_size+itr)+"_m")
                utils.save_image(pred_o[itr].astype(np.uint8), "../"+FLAGS.logs_dir + "trained_images/prediction", name="pred_" + str(i*FLAGS.batch_size+itr)+"_o")
                print("Saved image: %d" % (i*FLAGS.batch_size+itr))


if __name__ == "__main__":
    tf.app.run()
