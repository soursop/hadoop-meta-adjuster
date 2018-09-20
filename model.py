import tensorflow as tf
import pandas as pd
import numpy as np
from scipy import stats
import math

import utils as ut
from sklearn import metrics
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt


def _boxcox(features, lmbdas={}):
    """ Scaling features by box-cox
    :param features: DataFrame
    """
    exist = any(lmbdas)
    for name in features.columns:
        if exist:
            features[name] = stats.boxcox(features[name], lmbdas[name])
        else:
            features[name], lmbdas[name] = stats.boxcox(features[name])
    return features, lmbdas


def _drop_high_corr(features, rate=0.95):
    """ Drop features that have high correlation from each others
    :param features: DataFrame
    """
    # Create correlation matrix
    corr_matrix = features.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than rate
    to_drop = [column for column in upper.columns if any(upper[column] > rate)]

    # Drop features
    features.drop(features.columns[to_drop], axis=1)
    return features


def _imitate(_from, _to, pred):
    """
    :param _from: range from
    :param _to: range to
    :param pred: predication function
    :return: DataFrame(x), DataFrame(y)
    """
    x, y = [(i, pred(i)) for i in range(_from, _to)]
    pd.DataFrame(list(x), columns=["x"]), pd.DataFrame(list(y), columns=["y"])


def _batch_normalization_restore(x,
                                 mean,
                                 variance,
                                 offset,
                                 scale,
                                 variance_epsilon,
                                 name=None):
    with ops.name_scope(name, "batchnorm_restore", [x, mean, variance, scale, offset]):
        inv = math_ops.rsqrt(variance + variance_epsilon)
        if scale is not None:
            inv *= scale
        return (x - (offset - mean * inv if offset is not None else -mean * inv)) / inv


def _layer(features, lebals, denses):
    l = features
    for d in denses:
        l = tf.layers.Dense(d)(l)
    return tf.layers.Dense(lebals.shape[1])(l)



class Normalizer:
    def __init__(self, features, epsilon=1e-3):
        """ Change features values by normalized values
        :param features: Tensor
        :param hidden_unit: hidden unit size
        :param epsilon : epsilon value
        """

        self.scale = tf.Variable(tf.ones([features.get_shape()[-1]]))
        self.beta = tf.Variable(tf.zeros([features.get_shape()[-1]]))
        self.pop_mean = tf.Variable(tf.zeros([features.get_shape()[-1]]), trainable=False)
        self.pop_var = tf.Variable(tf.ones([features.get_shape()[-1]]), trainable=False)

        batch_mean, batch_var = tf.nn.moments(features, [0])
        train_mean = tf.assign(self.pop_mean, batch_mean)
        train_var = tf.assign(self.pop_var, batch_var)

        self.epsilon = epsilon
        with tf.control_dependencies([train_mean, train_var]):
            self.batch = tf.nn.batch_normalization(features, batch_mean, batch_var, self.beta, self.scale, self.epsilon)

    def apply(self, sess, x, x_data):
        # Estimate 'estimate data' value by normalized processing from train data
        return sess.run(tf.nn.batch_normalization(x, self.pop_mean, self.pop_var, self.beta, self.scale, self.epsilon), feed_dict={x: x_data})

    def restore(self, sess, x, x_data):
        return sess.run(_batch_normalization_restore(x, self.pop_mean, self.pop_var, self.beta, self.scale, self.epsilon), feed_dict={x: x_data})

    def run(self, sess, feed_dict):
        return sess.run(self.batch, feed_dict=feed_dict)


class Handler:
    def __init__(self, x, y):
        self.x_data = x
        self.y = y

    def apply(self, x):
        return x

    def scatter(self):
        ut.scatter(self.x_data)


class BoxCox(Handler):
    def __init__(self, x, y):
        x_data, self.lmbdas = _boxcox(x)
        Handler.__init__(self, x_data, y)

    def apply(self, x):
        return _boxcox(x, self.lmbdas)[0]


class Gradient:
    def __init__(self, handler):
        """
        :param x: DataFrame
        :param y: DataFrame
        """
        self.handler = handler
        self.x_size = len(handler.x_data.columns)
        self.y_size = len(handler.y.columns)

    def run(self, training_epochs, extract, hidden_unit=1, learning_rate=0.01):
        # tf Graph Input
        x = tf.placeholder(tf.float32, [None, self.x_size])
        y = tf.placeholder(tf.float32, [None, self.y_size])

        # Set model weights
        w = tf.Variable(tf.zeros([self.x_size, hidden_unit]))
        b = tf.Variable(tf.zeros([hidden_unit]))

        # Construct model
        predict = tf.matmul(x, w) + b

        # Minimize error
        error = tf.square(predict - y)
        cost = tf.reduce_mean(error)

        normalizer = Normalizer(x)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

        errors = []
        weight = []
        bias = []

        # Start training
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # ut.scatter(self.handler.x_data.join(self.handler.y))
            xn = pd.DataFrame(normalizer.run(sess, feed_dict={x: self.handler.x_data}), columns=self.handler.x_data.columns)

            for step in range(training_epochs):
                _, loss = sess.run([optimizer, error], feed_dict={x: xn, y: self.handler.y})
                errors.append(loss)
                weight = sess.run(w)
                bias = sess.run(b)

            # print(np.column_stack((self.handler.y, np.add(np.matmul(xn, weight), bias))))
            score_sklearn = metrics.mean_squared_error(np.add(np.matmul(xn, weight), bias), self.handler.y)
            print('MSE (sklearn): {0:f}'.format(score_sklearn))
            return extract(sess, normalizer, x, weight, bias)


class DNNGradient:
    def __init__(self, handler):
        """
        :param x: DataFrame
        :param y: DataFrame
        """
        self.handler = handler
        self.x_size = len(handler.x_data.columns)
        self.y_size = 1 if type(handler.y) == pd.Series else len(handler.y.columns)

    def run(self, expect, log_dir, learning_rate=0.01, hidden_unit=500, steps=500):

        # Start training
        with tf.Session() as sess:
            # self.run_by_regressor(sess, learning_rate, hidden_unit, steps, expect, log_dir)
            self.run_by_graph(sess, [hidden_unit, hidden_unit], log_dir, learning_rate)
            return 1

    def run_by_regressor(self, sess, learning_rate, hidden_unit, steps, expect, log_dir):

        # tf Graph Input
        x = tf.placeholder(tf.float32, [None, self.x_size])

        normalizer = Normalizer(x)
        optimizer = tf.train.AdamOptimizer(learning_rate)

        feature_cols = [tf.feature_column.numeric_column(k) for k in self.handler.x_data.columns]
        regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols
                                              , optimizer=optimizer
                                              , config=tf.estimator.RunConfig(log_step_count_steps=100)
                                              , hidden_units=[hidden_unit, hidden_unit, hidden_unit], model_dir=log_dir)

        # train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
        sess.run(tf.global_variables_initializer())

        # ut.scatter(self.handler.x_data.join(self.handler.y))
        xn = pd.DataFrame(normalizer.run(sess, feed_dict={x: self.handler.x_data}), columns=self.handler.x_data.columns)
        train_input_fn = tf.estimator.inputs.pandas_input_fn(
            x=xn, y=self.handler.y, batch_size=20, num_epochs=None, shuffle=True)

        regressor.train(input_fn=train_input_fn, steps=steps)

        test_input_fn = tf.estimator.inputs.pandas_input_fn(
            x=xn, y=self.handler.y, batch_size=1, num_epochs=1, shuffle=False)
        ev = regressor.evaluate(input_fn=test_input_fn, steps=1)
        loss_score = ev["loss"]
        print("Loss: {0:f}".format(loss_score))

        # y_predicted = np.array(list(p['predictions'] for p in predictions))
        # y_predicted = y_predicted.reshape(np.array(self.handler.y).shape)
        # plt.scatter(y_predicted, self.handler.y)
        # plt.show()
        #
        # score_sklearn = metrics.mean_squared_error(y_predicted, self.handler.y)
        # print('MSE (sklearn): {0:f}'.format(score_sklearn))

        test_data = pd.DataFrame(normalizer.run(sess, feed_dict={x: expect}), columns=self.handler.x_data.columns)
        test_input_fn = tf.estimator.inputs.pandas_input_fn(
            x=test_data, batch_size=1, num_epochs=1, shuffle=False)
        predictions = regressor.predict(input_fn=test_input_fn)
        y_predicted = np.array(list(p['predictions'] for p in predictions))


        plt.plot(y_predicted)
        plt.show()


    def run_by_graph(self, sess, hidden, log_dir, learning_rate=0.01, BATCH_SIZE=50, EPOCHS=200):
        x = tf.placeholder(tf.float32, [None, self.x_size])
        normalizer = Normalizer(x)

        sess.run(tf.global_variables_initializer())

        xn = normalizer.run(sess, feed_dict={x: self.handler.x_data})

        dataset = tf.data.Dataset.from_tensor_slices((xn, self.handler.y)).repeat().batch(BATCH_SIZE).shuffle(buffer_size=100)
        iter = dataset.make_one_shot_iterator()
        next = iter.get_next()

        def train_input_fn(
                features,  # This is batch_features from input_fn
                labels,  # This is batch_labels from input_fn
                mode):  # And instance of tf.estimator.ModeKeys, see below
            print(features)
            logits = _layer(features, labels, hidden)

            # Shape [4]
            predictions = tf.squeeze(logits)
            loss = tf.reduce_mean(tf.square(predictions - tf.to_float(labels)))
            tf.summary.scalar('loss', loss)

            average_loss = tf.losses.mean_squared_error(labels, predictions)
            tf.summary.scalar('average_loss', average_loss)

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(
                loss,
                global_step=tf.train.get_global_step())

            return loss, tf.estimator.EstimatorSpec(
                mode,
                loss=loss,
                train_op=train_op)

        regressor = tf.estimator.Estimator(
            model_fn=train_input_fn,
            model_dir=log_dir)

        for i in range(EPOCHS):
            regressor.train(input_fn=lambda: next)
        return 0

