# coding: utf-8
import sys 
import importlib
importlib.reload(sys)

from lib.config import cfg

import pickle
import pandas as pd
import tensorflow as tf

def open_pickle(pickle_file):
	
	f = open(pickle_file, 'rb')
	pick_data = pickle.load(f, encoding='bytes')
	return pick_data

def open_csv(csv_file):

	data = pd.read_csv(csv_file)
	return data

def compute_sequence_length(input_batch):

	with tf.variable_scope('seq_len'):
		used = tf.greater(input_batch, 0)
		seq_length = tf.reduce_sum(tf.cast(used, tf.int32), reduction_indices=1)
		seq_length = tf.cast(seq_length, tf.int32)

	return seq_length


def extract_last_output(output):

        out_size = int(output.get_shape()[2])

        index = tf.range(0, cfg.CONST.batch_size) * 16
        flat = tf.reshape(output, [-1, out_size])

        relevant = tf.gather(flat, index)
        return relevant