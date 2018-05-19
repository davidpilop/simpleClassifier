import os
from numpy import genfromtxt
import tensorflow as tf

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

def load_data(label_name='Species'):
    """Parses the csv file in TRAIN_URL and TEST_URL."""

    # Create a local copy of the training set.
    train_path = tf.keras.utils.get_file(fname=TRAIN_URL.split('/')[-1],
                                         origin=TRAIN_URL)

    # Parse the local CSV file.
    my_data = genfromtxt(TRAIN_URL, delimiter=',', skip_header=1)
    train_features = my_data[:,:-1]
    train_label = my_data[:,-1]
    return train_features, train_label

def num_labels():
    file = open('meta.txt', 'r')
    lines = file.readlines()
    for i,line in enumerate(lines):
        lines[i] = line.strip()
    return lines

if __name__ == "__main__":
    # Call load_data() to parse the CSV file.
    features, label = load_data()

    # View a single example entry from a batch
    print("example features:", features[0])
    print("example label:", label[0])

    # meta
    num_labels()
