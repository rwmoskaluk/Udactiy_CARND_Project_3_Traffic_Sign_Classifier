# Load pickled data
import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
from tensorflow.contrib.layers import flatten
import tensorflow as tf
from sklearn.utils import shuffle
import glob
import cv2


def extract_data():
    """
    Extract pickled data from local folder
    :return: extracted data in train, validate, test format
    """
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

    n_train = len(X_train)

    n_validation = len(X_valid)

    n_test = len(X_test)

    image_shape = np.shape(X_train[0])

    n_classes = len(np.unique(y_train))

    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Number of validation examples =", n_validation)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)

    return X_train, y_train, X_valid, y_valid, X_test, y_test, n_classes


def pre_process_data(image_data):
    """
    Image pre-processing step
    1) grayscale image from RGB
    2) Normalize image
    :param image_data: ndarray of image data in RGB colorspace
    :return:
    """
    # grayscale image
    if image_data.ndim == 3:
        image_data = np.sum(image_data / 3, axis=2, keepdims=True)
    else:
        image_data = np.sum(image_data / 3, axis=3, keepdims=True)
    # normalize image
    image_data = (image_data - 128) / 128
    return image_data


def model(X_train, y_train, X_valid, y_valid, X_test, y_test, x_german_test, y_german_test, n_classes,
          train, test, german):
    """
    Model function for training and testing a LeNet Model
    Dropout and L2 regularization is utilized
    :param y_german_test:
    :param x_german_test:
    :param german:
    :param X_train: Training samples (ndarray x 32 x 32 x 1)
    :param y_train: Training classes (ndarray x 43)
    :param X_valid: Validation samples (ndarray x 32 x 32 x 1)
    :param y_valid: Validation classes (ndarray x 43)
    :param X_test: Test samples
    :param y_test: Test classes
    :param n_classes: total # of classes in data
    :param train: train model flag
    :param test: test model flag
    :return:
    """

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

        # Flatten. Input = 5x5x16. Output = 400. then dropout
        fc0 = flatten(conv2)
        fc0 = tf.nn.dropout(fc0, keep_prob=keep_prob)

        # Layer 3: Fully Connected. Input = 400. Output = 120.
        fc1_w = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
        fc1_b = tf.Variable(tf.zeros(120))
        fc1 = tf.matmul(fc0, fc1_w) + fc1_b
        L2_loss1 = 0.001 * tf.nn.l2_loss(fc1_w)

        # Activation and then dropout
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob)

        # Layer 4: Fully Connected. Input = 120. Output = 84.
        fc2_w = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
        fc2_b = tf.Variable(tf.zeros(84))
        fc2 = tf.matmul(fc1, fc2_w) + fc2_b
        L2_loss2 = 0.001 * tf.nn.l2_loss(fc2_w)

        # Activation and then dropout
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
            test_accuracy = evaluate(X_test, y_test)
            print("Test Accuracy = {:.3f}".format(test_accuracy))

    if german is True:
        pred_operation = tf.argmax(logits, 1)
        softmax_logits = tf.nn.softmax(logits)
        top_k_prob = tf.nn.top_k(softmax_logits, 5)
        with tf.Session() as sess:
            model_dir = os.path.join(os.getcwd(), "model\\model.ckpt")
            saver.restore(sess, model_dir)
            german_predictions = sess.run(pred_operation, feed_dict={x: x_german_test, keep_prob: 1.0})
            print(german_predictions)
            german_accuracy = evaluate(x_german_test, y_german_test)
            print("Test Accuracy German Signs = {:.3f}".format(german_accuracy))
            top_k_prob = sess.run(top_k_prob, feed_dict={x: x_german_test, keep_prob: 1.0})
            print(top_k_prob)
            visualize_topk(top_k_prob, y_german_test)


def German_Signs():
    """
    Read in German signs found on Google maps in Berlin, Germany along Kronenstraße street
    :return: image data for testing and respective labels
    """
    images = glob.glob(os.path.join(os.getcwd(), "German_Signs\\") + "*.jpg")
    y_test = np.array([11, 15, 13, 26, 1])
    img_list = []
    for image in images:
        img = cv2.imread(image)
        img_list.append(img)

    x_test = np.stack(img_list, axis=0)
    x_test = pre_process_data(x_test)
    return x_test, y_test


def data_visualization(X_train, y_train, n_classes):
    """
    Visualize data set
    :param n_classes:
    :param y_train:
    :param X_train:
    :return:
    """
    plt.title('Traffic Sign Frequency of Labels, Training Set')
    plt.xlabel('Numeric Label')
    plt.ylabel('Frequency')
    plt.hist(y_train, bins=np.arange(y_train.min(), y_train.max()+1))
    plt.savefig('visualizations/Label_Frequency.png', bbox_inches='tight')

    filename = 'signnames.csv'
    with open(filename, 'rb') as f:
        label_list = f.readlines()[1:]

    cols = 6
    rows = 7

    ax = []
    fig = plt.figure(figsize=(32, 32))
    # fig.tight_layout()

    for loc in range(0, n_classes-1):
        # array_indices = np.where(y_train == label)
        array_indices = np.argwhere(y_train == loc)
        ax.append(fig.add_subplot(rows, cols, loc+1))
        ax[-1].set_title(label_list[loc].decode("utf-8"), fontsize=20)
        plt.imshow(X_train[array_indices[50][0]])

    fig.subplots_adjust(hspace=0.5)
    plt.savefig('visualizations/Training_Visualization_Images.png', bbox_inches='tight')

    fig = plt.figure(figsize=(32, 32))
    ax = []
    rows = 1
    cols = 2
    ax.append(fig.add_subplot(rows, cols, 1))
    ax[-1].set_title('Original Image')
    plt.imshow(X_train[2050])
    ax.append(fig.add_subplot(rows, cols, 2))
    ax[-1].set_title('Normalized and Grayscale Image')
    gray_scale = pre_process_data(X_train[2050])
    plt.imshow(gray_scale.reshape(32, 32), cmap='gray')

    plt.savefig('visualizations/Traffic_Sign_Grayscale.png', bbox_inches='tight')
    plt.show()


def visualize_topk(top_k_prob, y_german_test):
    """
    Visualize German signs with top 5 predicted probabilities for sign classification
    :return:
    """
    filename = 'signnames.csv'
    with open(filename, 'rb') as f:
        label_list = f.readlines()[1:]
    ax = []
    rows = 5
    cols = 1
    fig = plt.figure(figsize=(30, 30))
    fig.subplots_adjust(hspace=1.0)
    fig.tight_layout()
    for i in range(0, rows):
        ax.append(fig.add_subplot(rows, cols, i+1))
        ax[-1].set_title(label_list[y_german_test[i]].decode("utf-8"), fontsize=20)
        ax[-1].barh([10, 30, 50, 70, 90], top_k_prob[0][i], align='center', height=7)
        sub_label_list = [label_list[j].decode("utf-8") for j in top_k_prob[1][i]]
        plt.yticks([10, 30, 50, 70, 90], sub_label_list)
        ax[-1].tick_params(axis='both', which='major', labelsize=20)
        ax[-1].tick_params(axis='both', which='minor', labelsize=20)
        # yminorlocator = plt.MaxNLocator(nbins=5)
        # ax[-1].yaxis.set_minor_locator(yminorlocator)

    plt.savefig('visualizations/Top_5_K.png', bbox_inches='tight')
    plt.show()


def main():

    X_train, y_train, X_valid, y_valid, X_test, y_test, n_classes = extract_data()
    # data_visualization(X_train, y_train, n_classes)
    X_train = pre_process_data(X_train)
    X_valid = pre_process_data(X_valid)
    X_test = pre_process_data(X_test)
    x_test_german, y_test_german = German_Signs()
    model(X_train, y_train, X_valid, y_valid, X_test, y_test, x_test_german, y_test_german,
          n_classes, train=False, test=True, german=True)

if __name__ == "__main__":
    main()
