import tensorflow as tf
import numpy as np
import pandas as pd

label_num = 7  # 7 is output labels size
attr_num = 8  # 8 is the attrs size, without the 'TIME'

batch_size = 128
num_step = 3000

def init_dataset():
    """Initialize the datasets."""

    _train_data_set = pd.read_csv(r'./dataset/shuttle.trn', sep=' ').astype('float32')
    _cv_data_set = pd.read_csv(r'./dataset/shuttle_cv.trn', sep=' ').astype('float32')
    _test_data_set = pd.read_csv(r'./dataset/shuttle_test.trn', sep=' ').astype('float32')
    del _train_data_set['TIME']
    del _cv_data_set['TIME']
    del _test_data_set['TIME']

    _train_label_set = _train_data_set['CLASS']
    _train_label_set = (np.arange(7) == _train_label_set[:, None]).astype(np.float32) #label one-hot
    del _train_data_set['CLASS']

    _cv_label_set = _cv_data_set['CLASS']
    _cv_label_set = (np.arange(7) == _cv_label_set[:, None]).astype(np.float32)  # label one-hot
    del _cv_data_set['CLASS']

    _test_label_set = _test_data_set['CLASS']
    _test_label_set = (np.arange(7) == _test_label_set[:, None]).astype(np.float32)  # label one-hot
    del _test_data_set['CLASS']

    print('Training: ', _train_data_set.shape, _train_label_set.shape)
    print('CV: ', _cv_data_set.shape, _cv_label_set.shape)
    print('Test: ', _test_data_set.shape, _test_label_set.shape)

    return _train_data_set, _train_label_set, _cv_data_set, _cv_label_set, _test_data_set, _test_label_set


def accuracy(prediction, labels):
    """Calculate the accuracy of the model."""
    return (100.0 * np.sum(np.argmax(prediction, 1) == np.argmax(labels, 1))) / (prediction.shape[0])


def nn_diagram_define(_cv_data_set, _test_data_set):
    """Define the tensorflow diagram architecture."""

    _graph = tf.Graph()
    with _graph.as_default():
        _tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, attr_num))
        _tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, label_num))
        _lambda_regular = tf.placeholder(tf.float32)  # regularization rate λ

        tf_cv_dataset = tf.constant(_cv_data_set)
        tf_test_dataset = tf.constant(_test_data_set)

        weights = tf.Variable(tf.truncated_normal([attr_num, label_num]))  # random initialize the weight
        biases = tf.Variable(tf.zeros([label_num]))

        logits = tf.matmul(_tf_train_dataset, weights) + biases  # softmax layer
        _loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=_tf_train_labels, logits=logits)) + \
            _lambda_regular * tf.nn.l2_loss(weights)

        _optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(_loss)

        _train_prediction = tf.nn.softmax(logits)
        _cv_prediction = tf.nn.softmax(tf.matmul(tf_cv_dataset, weights) + biases)
        _test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

        return _graph, _optimizer, _loss, weights, biases, _train_prediction, _cv_prediction, _test_prediction, _tf_train_dataset, _tf_train_labels, _lambda_regular


def nn_process_diagram(_graph, _optimizer, _loss, weights, biases,_train_prediction, _cv_prediction, _cv_label_set, _test_prediction, _test_label_set,
                       _train_data_set, _train_label_set, _tf_train_dataset, _tf_train_labels, _lambda_regular):
    """Process the tensorflow diagram."""

    with tf.Session(graph=_graph) as session:
        tf.global_variables_initializer().run()
        print('Initialized')
        for step in range(num_step):
            offset = (step * batch_size) % (_train_label_set.shape[0] - batch_size) #128 256 ...
            batch_data = _train_data_set[offset:(offset + batch_size)]
            batch_label = train_label_set[offset:(offset + batch_size)]
            feed_dict = {_tf_train_dataset : batch_data, _tf_train_labels : batch_label, _lambda_regular : 1e-2}  #0.001
            _, l, predictions = session.run([_optimizer, _loss, _train_prediction], feed_dict=feed_dict)

            if (step % 500 == 0) :
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_label))
                print("Validation accuracy: %.1f%%" % accuracy(
                    _cv_prediction.eval(), _cv_label_set))
        print("Weights: ", weights.eval())
        print("Biases: ", biases.eval())
        print("Test accuracy: %.1f%%" % accuracy(_test_prediction.eval(), _test_label_set))

if __name__ == '__main__':
    train_data_set, train_label_set, cv_data_set, cv_label_set, \
    test_data_set, test_label_set \
        = init_dataset()


    graph, optimizer, loss, weights, biases, train_prediction, cv_prediction, \
    test_prediction, tf_train_dataset, tf_train_labels, \
    lambda_regular \
        = nn_diagram_define(cv_data_set,
                            test_data_set)

    nn_process_diagram(graph, optimizer, loss, weights, biases, train_prediction,
                       cv_prediction, cv_label_set, test_prediction, test_label_set, train_data_set,
                       train_label_set, tf_train_dataset, tf_train_labels,
                       lambda_regular)


    # nn_diagram_define(dataset=dataset)
