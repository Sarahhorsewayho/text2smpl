import os
import sys
import math
import numpy as np
import tensorflow as tf 

import lib.load_data as ld
import lib.caption_process as cp

from lib.config import cfg
from lib.model import Model
from lib.utils import open_csv
from lib.utils import open_pickle
from lib.optimiser import Optimiser
from lib.load_data import Preprocessor
from lib.write_output import write_to_mesh
from tensorflow.python import pywrap_tensorflow

def main():

	f = open('/home/harryh/t2b/logger.log', 'w')
	sys.stderr = f 

	checkpoint_path = '/home/harryh/t2b/data/Model-20001'

	smplparams_mean = np.load(cfg.DIR.smplparams_mean)
	smplparams_std = np.load(cfg.DIR.smplparams_std)

	param_selection = []
	param_selection += range(10)
	param_selection += range(10, 226)

	latent_mean = smplparams_mean[param_selection]
	latent_std = smplparams_std[param_selection]

	input_dict = open_csv(cfg.DIR.test_caption_path)
	description = input_dict['description']
	caption_post = cp.gen_captions(description)

	model_inputs = tf.placeholder(tf.float64, shape=[cfg.CONST.batch_size, 16, 128])
	model = Model(model_inputs,
				  tf.constant(latent_mean, dtype=np.float64),
				  tf.constant(latent_std, dtype=np.float64),
				  is_training=True)

	outputs = model.get_outputs()

	saver = tf.train.Saver()

	with tf.Session() as sess:
		saver.restore(sess, checkpoint_path)
		initializer = tf.global_variables_initializer()
		sess.run(initializer)
		predicted = sess.run(outputs, feed_dict={model_inputs: caption_post})

	write_to_mesh(predicted, latent_mean, latent_std)
	

if __name__ == '__main__':
	main()