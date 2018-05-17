import os
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

import preprocess
import model

LOG_DIR = './log'
NUM_EPOCHS = 201
BATCH_SIZE = 64
NUM_FEATURES = 4
learning_rate = 0.01
os.system('mkdir {}'.format(LOG_DIR))

def train():
    with tf.Graph().as_default():
        features, labels = model.placeholder_inputs(BATCH_SIZE, NUM_FEATURES)

        pred = model.get_model(features)
        with tf.name_scope('loss') as scope:
            loss = model.get_loss(pred, labels)
        tf.summary.scalar('loss', loss)

        with tf.name_scope('metrics') as scope:
            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_FEATURES)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            optimizer = tf.train.AdamOptimizer(learning_rate)
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
        sess.run(init)

        for epoch in range(NUM_EPOCHS):
            data, label = preprocess.load_data()
            file_size = data.shape[0]
            num_batches = file_size // BATCH_SIZE

            for batch_idx in tqdm(range(num_batches), ncols= 100, desc = str(epoch)):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = (batch_idx+1) * BATCH_SIZE

                feed_dict = {features: data[start_idx:end_idx, :],
                             labels: label[start_idx:end_idx]}
                summary, loss_val, pred_val = sess.run([merged, loss, pred],feed_dict=feed_dict)
                train_writer.add_summary(summary, batch_idx)

            save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
    return

if __name__ == "__main__":
    train()