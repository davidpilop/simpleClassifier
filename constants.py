#!/usr/bin/env python
# encoding: utf-8
"""
Constants.py
"""

LOG_DIR = './log'
NUM_EPOCHS = 201
BATCH_SIZE = 120
NUM_FEATURES = 4
learning_rate = 0.01
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
