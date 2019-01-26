# Load pickled data
import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
from tensorflow.contrib.layers import flatten
import tensorflow as tf
from sklearn.utils import shuffle


# TODO: Fill this in based on where you saved the training and testing data
def extract_data():

    training_file = 'data/train.p'
    validation_file = 'data/valid.p'
    testing_file = 'data/test.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']

    # TODO: Number of training examples
    n_train = len(X_train)

    # TODO: Number of validation examples
    n_validation = len(X_valid)

    # TODO: Number of testing examples.
    n_test = len(X_test)

    # TODO: What's the shape of an traffic sign image?
    image_shape = np.shape(X_train[0])

    # TODO: How many unique classes/labels there are in the dataset.
    n_classes = len(np.unique(y_train))

    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)

    return X_train, y_train, X_valid, y_valid, X_test, y_test, n_classes

# TODO: Data visualization
# Data exploration visualization step
# Data exploration visualization code goes here.
# Feel free to use as many code cells as needed.


def data_visualization(data):
    pass
# plt.imshow(X_train[0])
# plt.show()

# TODO: Preprocessing step (Normalization and grayscale conversion)
# Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include
# converting to grayscale, etc.
# Feel free to use as many code cells as needed.


def pre_process_data(image_data):
    # grayscale image
    image_data = np.sum(image_data / 3, axis=3, keepdims=True)
    # normalize image
    image_data = (image_data - 128) / 128
    return image_data

# TODO: Definition of Architectural step
# Define your architecture here.


def model(X_train, y_train, X_valid, y_valid, X_test, y_test, n_classes, train, test):
    # TODO: Train model step
    # Train your model here.
    # Calculate and report the accuracy on the training and validation set.
    # Once a final model architecture is selected,
    # the accuracy on the test set should be calculated and reported as well.
    # Feel free to use as many code cells as needed.

    EPOCHS = 100
    BATCH_SIZE = 128
    rate = 0.0005
    drop_value = 0.5

    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    y = tf.placeholder(tf.int32, None)
    one_hot_y = tf.one_hot(y, n_classes)
    keep_prob = tf.placeholder(tf.float32)

    def lenet(x_data):
        # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
        mu = 0
        sigma = 0.1

        # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
        conv1_w = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma))
        conv1_b = tf.Variable(tf.zeros(6))
        conv1 = tf.nn.conv2d(x_data, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

        # Activation.
        conv1 = tf.nn.relu(conv1)

        # Pooling. Input = 28x28x6. Output = 14x14x6.
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # Layer 2: Convolutional. Output = 10x10x16.
        conv2_w = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
        conv2_b = tf.Variable(tf.zeros(16))
        conv2 = tf.nn.conv2d(conv1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

        # Activation.
        conv2 = tf.nn.relu(conv2)

        # Pooling. Input = 10x10x16. Output = 5x5x16.
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # Flatten. Input = 5x5x16. Output = 400.
        fc0 = flatten(conv2)
        fc0 = tf.nn.dropout(fc0, keep_prob=keep_prob)

        # Layer 3: Fully Connected. Input = 400. Output = 120.
        fc1_w = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
        fc1_b = tf.Variable(tf.zeros(120))
        fc1 = tf.matmul(fc0, fc1_w) + fc1_b
        L2_loss1 = 0.001 * tf.nn.l2_loss(fc1_w)

        # Activation.
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob)

        # Layer 4: Fully Connected. Input = 120. Output = 84.
        fc2_w = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
        fc2_b = tf.Variable(tf.zeros(84))
        fc2 = tf.matmul(fc1, fc2_w) + fc2_b
        L2_loss2 = 0.001 * tf.nn.l2_loss(fc2_w)

        # Activation.
        fc2 = tf.nn.relu(fc2)
        fc2 = tf.nn.dropout(fc2, keep_prob=keep_prob)

        # Layer 5: Fully Connected. Input = 84. Output = 43.
        fc3_w = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma))
        fc3_b = tf.Variable(tf.zeros(43))
        logits = tf.matmul(fc2, fc3_w) + fc3_b
        L2_loss3 = 0.001 * tf.nn.l2_loss(fc3_w)

        L2_loss = L2_loss1 + L2_loss2 + L2_loss3

        return logits, L2_loss

    logits, L2_loss = lenet(x)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_y, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy + L2_loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=rate)
    training_operation = optimizer.minimize(loss_operation)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()

    def evaluate(X_data, y_data):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
            accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples

    if train is True:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            num_examples = len(X_train)

            print("Training...")
            print()
            for i in range(EPOCHS):
                X_train, y_train = shuffle(X_train, y_train)
                for offset in range(0, num_examples, BATCH_SIZE):
                    end = offset + BATCH_SIZE
                    batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                    sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: drop_value})

                validation_accuracy = evaluate(X_valid, y_valid)
                training_accuracy = evaluate(X_train, y_train)
                print("EPOCH {} ...".format(i+1))
                print("Validation Accuracy = {:.3f}".format(validation_accuracy))
                print("Training Accuracy = {:.3f}".format(training_accuracy))
                print()
            model_dir = os.path.join(os.getcwd(), "model\\model.ckpt")
            saver.save(sess, model_dir)
            print("Model saved")

    if test is True:
        with tf.Session() as sess:
            model_dir = os.path.join(os.getcwd(), "model\\model.ckpt")
            saver.restore(sess, model_dir)
            print('')

            def evaluate(X_data, y_data):
                num_examples = len(X_data)
                total_accuracy = 0
                sess = tf.get_default_session()
                for offset in range(0, num_examples, BATCH_SIZE):
                    batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
                    accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                    total_accuracy += (accuracy * len(batch_x))
                return total_accuracy / num_examples
            test_accuracy = evaluate(X_test, y_test)
            print("Test Accuracy = {:.3f}".format(test_accuracy))


def main():

    X_train, y_train, X_valid, y_valid, X_test, y_test, n_classes = extract_data()
    X_train = pre_process_data(X_train)
    X_valid = pre_process_data(X_valid)
    X_test = pre_process_data(X_test)
    model(X_train, y_train, X_valid, y_valid, X_test, y_test, n_classes, train=False, test=True)

if __name__ == "__main__":
    main()
