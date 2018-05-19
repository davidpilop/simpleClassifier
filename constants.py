#!/usr/bin/env python
# encoding: utf-8
"""
Constants.py
"""

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--max_epoch', type=int, default=200, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=120, help='Batch Size during training [default: 24]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_EPOCHS = FLAGS.max_epoch
LEARNING_RATE = FLAGS.learning_rate
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
NUM_FEATURES = 4
