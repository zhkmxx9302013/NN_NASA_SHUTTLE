import tensorflow as tf
import numpy as np
import pandas as pd

is_train_option = True
#True #global training identifier. Used for judging whether this is a training process or not.

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


# def accuracy(prediction, labels):
#     """Calculate the accuracy of the model."""
#     return (100.0 * np.sum(np.argmax(prediction, 1) == np.argmax(labels, 1))) / (prediction.shape[0])


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)



def nn_diagram_define(_cv_data_set, _test_data_set, _train_label_set, _cv_label_set, _test_label_set):
    """Define the tensorflow diagram architecture."""

    _graph = tf.Graph()
    with _graph.as_default():
        _tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, attr_num))
        _tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, label_num))
        _lambda_regular = tf.placeholder(tf.float32)  # regularization rate λ

        tf_cv_dataset = tf.constant(_cv_data_set)
        tf_test_dataset = tf.constant(_test_data_set)

        with tf.name_scope('Layer_1'):
            with tf.name_scope('weights'):
                weights = tf.Variable(tf.truncated_normal([attr_num, label_num]))  # random initialize the weight
                variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = tf.Variable(tf.zeros([label_num]))
                variable_summaries(biases)
            with tf.name_scope('Wx_Plus_b'):
                logits = tf.matmul(_tf_train_dataset, weights) + biases  # softmax layer
                tf.summary.histogram('logits', logits)
            with tf.name_scope('cross_entropy'):
                _loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=_tf_train_labels, logits=logits)) + \
                    _lambda_regular * tf.nn.l2_loss(weights)
            tf.summary.scalar('cross_entropy', _loss)
            with tf.name_scope('train'):
                _optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(_loss)

        _train_prediction = tf.nn.softmax(logits)
        _cv_prediction = tf.nn.softmax(tf.matmul(tf_cv_dataset, weights) + biases)
        _test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

        with tf.name_scope('train_accuracy'):
            with tf.name_scope('train_correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(_train_prediction, 1), tf.argmax(_tf_train_labels, 1))
            with tf.name_scope('train_accuracy'):
                _train_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))* 100.0
        tf.summary.scalar('train_accuracy', _train_accuracy)

        with tf.name_scope('cv_accuracy'):
            with tf.name_scope('cv_correct_prediction'):
                cv_correct_prediction = tf.equal(tf.argmax(_cv_prediction, 1), tf.argmax(_cv_label_set, 1))
            with tf.name_scope('cv_accuracy'):
                _cv_accuracy = tf.reduce_mean(tf.cast(cv_correct_prediction, tf.float32))* 100.0
        tf.summary.scalar('cv_accuracy', _cv_accuracy)

        with tf.name_scope('test_accuracy'):
            with tf.name_scope('test_correct_prediction'):
                test_correct_prediction = tf.equal(tf.argmax(_test_prediction, 1), tf.argmax(_test_label_set, 1))
            with tf.name_scope('test_accuracy'):
                _test_accuracy = tf.reduce_mean(tf.cast(test_correct_prediction, tf.float32)) * 100.0
        tf.summary.scalar('test_accuracy', _test_accuracy)

        _saver = tf.train.Saver()

        return _test_accuracy, _cv_accuracy, _train_accuracy, _saver, _graph, _optimizer, _loss, weights, biases, _train_prediction, _cv_prediction, _test_prediction, _tf_train_dataset, _tf_train_labels,tf_test_dataset ,_lambda_regular


def nn_process_diagram(_test_accuracy, _cv_accuracy, _train_accuracy, _saver, _graph, _optimizer, _loss, weights, biases,_train_prediction, _cv_prediction, _cv_label_set, _test_prediction, _test_label_set,
                       _train_data_set, _train_label_set, _tf_train_dataset, _tf_train_labels,_tf_test_dataset , _lambda_regular):
    """Process the tensorflow diagram."""

    checkpoint_dir = './checkpoint/'
    summaries_dir = './summaries/'

    with tf.Session(graph=_graph) as session:
        merged_summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(summaries_dir + '/train', session.graph)

        tf.global_variables_initializer().run()

        if is_train_option:  # Training process.
            print('Initialized')
            for step in range(num_step):
                offset = (step * batch_size) % (_train_label_set.shape[0] - batch_size) #128 256 ...
                batch_data = _train_data_set[offset:(offset + batch_size)]
                batch_label = _train_label_set[offset:(offset + batch_size)]
                feed_dict = {_tf_train_dataset : batch_data, _tf_train_labels : batch_label, _lambda_regular : 2e-2}  #0.001
                test_acc, cv_acc, train_acc, summary, optim, l, predictions = session.run([_test_accuracy, _cv_accuracy, _train_accuracy, merged_summary_op, _optimizer, _loss, _train_prediction], feed_dict=feed_dict)

                if step % 500 == 0:
                    print("Minibatch loss at step %d: %f" % (step, l))
                    print("Minibatch accuracy: %.1f%%" % test_acc)
                    print("Validation accuracy: %.1f%%" % cv_acc)
                    _saver.save(session, checkpoint_dir + 'buaann.ckpt', global_step=int(step/500))

                train_writer.add_summary(summary, step)
            print("Weights: ", weights.eval())
            print("Biases: ", biases.eval())
            print(_test_prediction.eval())
            print("Test accuracy: %.1f%%" % test_acc)
        else:  # Using model process.
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                _saver.restore(session, ckpt.model_checkpoint_path)
            else:
                pass
            print(session.run(weights))
            print(session.run(biases))
            test_prediction = tf.nn.softmax(tf.matmul(_tf_test_dataset, weights) + biases)
            print(np.argmax(test_prediction.eval(), 1))


if __name__ == '__main__':
    train_data_set, train_label_set, cv_data_set, cv_label_set, \
    test_data_set, test_label_set \
        = init_dataset()


    test_accuracy, cv_accuracy,train_accuracy, saver, graph, optimizer, loss, weights, biases, train_prediction, cv_prediction, \
    test_prediction, tf_train_dataset, tf_train_labels, tf_test_dataset, \
    lambda_regular \
        = nn_diagram_define(cv_data_set,
                            test_data_set, train_label_set, cv_label_set, test_label_set)

    nn_process_diagram(test_accuracy, cv_accuracy, train_accuracy, saver, graph, optimizer, loss, weights, biases, train_prediction,
                       cv_prediction, cv_label_set, test_prediction, test_label_set, train_data_set,
                       train_label_set, tf_train_dataset, tf_train_labels, tf_test_dataset,
                       lambda_regular)


    # nn_diagram_define(dataset=dataset)
