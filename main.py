import os
import sys
import math
import numpy as np
import tensorflow as tf 

import lib.load_data as ld
import lib.caption_process as cp

from lib.config import cfg
from lib.model import Model
from lib.optimiser import Optimiser
from lib.load_data import Preprocessor
from lib.utils import open_csv
from lib.write_output import write_to_mesh

def main():

	f = open('/home/harryh/t2b/logger.log', 'w')
	sys.stderr = f 

	smplparams_mean = np.load(cfg.DIR.smplparams_mean)
	smplparams_std = np.load(cfg.DIR.smplparams_std)

	param_selection = []
	param_selection += range(10, 226)

	latent_mean = smplparams_mean[param_selection]
	latent_std = smplparams_std[param_selection]

	preprocessor = Preprocessor(latent_mean, latent_std)

	num_of_data = preprocessor.get_num_data()
	batch_data = preprocessor.batch_of_data

	steps_per_epoch = int(math.ceil(1.0 * num_of_data / cfg.CONST.batch_size))

	model_inputs = batch_data['embeddings']
	targets = batch_data['body_params']

	test_dict = open_csv(cfg.DIR.test_caption_path)
	description = test_dict['description']

	caption_post = cp.gen_captions(description)

	model = Model(model_inputs,
				  tf.constant(latent_mean, dtype=np.float64),
				  tf.constant(latent_std, dtype=np.float64),
				  is_training=True)

	opt = Optimiser(model,
					targets,
					tf.constant(latent_mean, dtype=np.float64),
				  	tf.constant(latent_std, dtype=np.float64))

	opt.prepare_loss_ops()
	loss_full = opt.get_loss_op()

	global_step = tf.Variable(
		name="global_step",
		expected_shape=(),
		dtype=tf.int64,
		trainable=False,
		initial_value=0)

	saver = tf.train.Saver(max_to_keep=150)

	with tf.Session() as sess:
		max_steps = steps_per_epoch * cfg.CONST.max_epochs
		print("steps_per_epoch: " + str(steps_per_epoch) + " max_epochs: " + str(cfg.CONST.max_epochs) + " max_steps: " + str(max_steps))
		total_examples_presented = num_of_data * cfg.CONST.max_epochs
		train_op = opt.prepare_train_op(global_step, max_steps)
		outputs = model.get_outputs()
		writer = tf.summary.FileWriter("logs/", sess.graph)
		initializer = tf.global_variables_initializer()
		sess.run(initializer)
		step = 0
		preprocessor.initialise_iterator(sess)

		try:
			while step <= max_steps:
				fetch = {"train_op": train_op, "loss": loss_full, "learning_rate": opt.learning_rate, "global_step": global_step}
				results = sess.run(fetch)
				step += 1
				if (step % 100 == 0):
					print("step: " + str(step) + " RESULTS: " + str(results))
				if (step % 50000 == 0):
					saver.save(sess,os.path.join(cfg.DIR.checkpoint_path,str(results['global_step']),"Model"),global_step=global_step)
				if(results['loss'] < 5e-15) : 
					break
				#if (results['loss'] < 0.13):
				#	break
		except tf.errors.OutOfRangeError:
			pass
		saver.save(sess,os.path.join(cfg.DIR.checkpoint_path,str(results['global_step']),"Model"),global_step=global_step)
		predicted = sess.run(outputs, feed_dict={model_inputs: caption_post})
		write_to_mesh(predicted, latent_mean, latent_std)
		print("shut down.")

if __name__ == '__main__':
	main()