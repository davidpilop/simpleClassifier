import tensorflow as tf
import numpy as np
import time

def placeholder_inputs(batch_size, num_features):
    features = tf.placeholder(tf.float32, shape=(batch_size, num_features), name='features')
    labels = tf.placeholder(tf.int32, shape=(batch_size), name='labels')
    return features, labels

def get_model(features):
    # input_data = tf.expand_dims(features, -1)
    w1 = tf.contrib.layers.fully_connected(
            inputs=features,
            num_outputs = 10,
            activation_fn=tf.nn.elu,
            normalizer_fn=None,
            normalizer_params=None,
            # weights_initializer=initializers.xavier_initializer(),
            # weights_regularizer=None,
            # biases_initializer=tf.zeros_initializer(),
            # biases_regularizer=None,
            # reuse=None,
            # variables_collections=None,
            # outputs_collections=None,
            # trainable=True,
            scope='hidden_1')
    w2 = tf.contrib.layers.fully_connected(
            inputs=w1,
            num_outputs = 10,
            activation_fn=tf.nn.elu,
            normalizer_fn=None,
            normalizer_params=None,
            # weights_initializer=initializers.xavier_initializer(),
            # weights_regularizer=None,
            # biases_initializer=tf.zeros_initializer(),
            # biases_regularizer=None,
            # reuse=None,
            # variables_collections=None,
            # outputs_collections=None,
            # trainable=True,
            scope='hidden_2')
    net = tf.contrib.layers.fully_connected(
            inputs=w2,
            num_outputs = 3,
            activation_fn=tf.nn.elu,
            normalizer_fn=None,
            normalizer_params=None,
            # weights_initializer=initializers.xavier_initializer(),
            # weights_regularizer=None,
            # biases_initializer=tf.zeros_initializer(),
            # biases_regularizer=None,
            # reuse=None,
            # variables_collections=None,
            # outputs_collections=None,
            # trainable=True,
            scope='final')
    return net

def get_loss(pred, label):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    return tf.reduce_mean(loss)

if __name__ == "__main__":
    with tf.Graph().as_default():
        batch_size = 100
        num_features = 4
        features, labels = placeholder_inputs(batch_size, num_features)
        net = get_model(features)
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            start = time.time()
            for i in range(100):
                print(i)
                sess.run(net, feed_dict={features: np.random.rand(batch_size, num_features)})
                print(sess.run(net, feed_dict={features: np.random.rand(batch_size, num_features)}))
            print(time.time() - start)
