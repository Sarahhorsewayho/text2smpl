import os
import random
import numpy as np
import tensorflow as tf

import lib.caption_process as cp

from lib.config import cfg
from lib.utils import open_csv
from lib.utils import open_pickle
from lib.conversions import aar_to_rotmat, prepare_kintree

class Preprocessor():

	def __init__(self, latent_mean=None, latent_std=None):

		self.latent_mean = latent_mean
		self.latent_std = latent_std
		
		self.input_dict = self.get_input_dict()
		self.num = self.get_num_data()

		self.orig_dict = {
			'embeddings': tf.placeholder(tf.float64, shape = [self.num, cfg.CONST.max_length, cfg.CONST.vec_dim]),
			'body_params': tf.placeholder(tf.float64, shape = [self.num, cfg.CONST.orig])
		}

		dataset = tf.data.Dataset.from_tensor_slices(self.orig_dict)
		#dataset = dataset.map(self.transform_data, num_parallel_calls=8)
		self.dataset = dataset.shuffle(cfg.CONST.shuffle_size)
		self.dataset = self.dataset.batch(cfg.CONST.batch_size, drop_remainder=True)
		self.dataset = self.dataset.repeat(cfg.CONST.max_epochs)	
		
		self.iterator = tf.data.Iterator.from_structure(self.dataset.output_types, self.dataset.output_shapes)
		self.batch_of_data = self.iterator.get_next()
		self.iter_init = self.iterator.make_initializer(self.dataset)

		return

	def load_data(self):
		data_dict = self.get_captions()
		self.caption_tuples = data_dict['description']
		self.body_params = np.zeros((len(data_dict), cfg.CONST.orig), dtype=np.float64)

		for i in range(len(data_dict['body_id'])):
			idx = data_dict['body_id'][i]
			self.body_params[i] = self.get_body_params(idx)

		cp.word_to_vec(self.caption_tuples)
		self.embeddings = cp.gen_captions(self.caption_tuples)
		self.embeddings = np.array(self.embeddings, dtype=np.float64)
		return self.embeddings, self.body_params

	def initialise_iterator(self, session):
		session.run(self.iter_init, feed_dict={self.orig_dict['embeddings']: self.input_dict['embeddings'],
														  self.orig_dict['body_params']: self.input_dict['body_params']})
		return

	def get_input_dict(self):
		self.data_list = self.get_data_list()
		self.embeddings, self.body_params = self.load_data()
		input_dict = {}
		input_dict['embeddings'] = self.embeddings
		input_dict['body_params'] = self.body_params
		return input_dict

	def get_data_list(self):	
		input_dict = self.get_captions()
		data_list = []
		
		for i in range(len(input_dict)):
			data_list.append(i)
		return data_list

	def transform_data(self, x):
		kintree = prepare_kintree()

		smplparams_orig = x['body_params']
		embedding = x['embeddings']

		smplparams_pose = tf.reshape(smplparams_orig, tf.convert_to_tensor((cfg.CONST.orig,), dtype=tf.int32))

		aar_convert = lambda i: aar_to_rotmat(i, kintree) 
		smplparams_pose = tf.py_func(aar_convert, [smplparams_pose], tf.float64)
		smplparams = tf.zeros([0,], dtype=tf.float64)
		smplparams = tf.concat([smplparams, smplparams_pose ], 0)

		smplparams = tf.reshape(smplparams, tf.convert_to_tensor((cfg.CONST.nz,), dtype=tf.int32))
		smplparams = tf.cast(smplparams, tf.float64)
		if self.latent_mean is not None:
			smplparams = tf.subtract(smplparams, self.latent_mean)
			if False and self.latent_std is not None:
				smplparams = tf.div(smplparams, self.latent_std)
				
		x['embeddings'] = embedding
		x['body_params'] = smplparams
		return x

	def get_num_data(self):
		return len(self.data_list)
		
	def get_captions(self):
		self.input_dict = open_csv(cfg.DIR.train_caption_path)
		return self.input_dict

	def get_body_params(self, body_id):
		tup = []
		body_params = open_pickle(os.path.join(cfg.DIR.body_path, body_id))
		pose_params = np.array(body_params[b'pose'], dtype=np.float64)
		return pose_params


