import os
import tensorflow as tf
from tqdm import tqdm
import numpy as np

import preprocess
import model
from constants import *

def train():
    with tf.Graph().as_default():
        features, labels = model.placeholder_inputs(BATCH_SIZE, NUM_FEATURES)

        pred = model.get_model(features)
        # with tf.name_scope('loss') as scope:
        loss = model.get_loss(pred, labels)
        tf.summary.scalar('loss', loss)

        total, count = tf.metrics.accuracy(labels=tf.to_int64(labels),
                                           predictions=tf.argmax(pred, 1),
                                           name='accuracy')
        tf.summary.scalar('accuracy', count)

        # Get training operator
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        train_op = optimizer.minimize(loss)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        # Create a session
        sess = tf.Session()

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        local = tf.local_variables_initializer()
        sess.run(init)
        sess.run(local)

        for epoch in range(NUM_EPOCHS):
            data, label = preprocess.load_data()

            feed_dict = {features: data,
                         labels: label}
            summary, _, loss_val, pred_val, accurate = sess.run([merged, train_op, loss, pred, count],feed_dict=feed_dict)
            train_writer.add_summary(summary, epoch)
            print(accurate)

            save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
    return

if __name__ == "__main__":
    train()
