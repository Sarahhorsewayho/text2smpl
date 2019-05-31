import numpy as np 
import tensorflow as tf 
import tensorflow.contrib.slim as slim

from lib.config import cfg
from lib.utils import compute_sequence_length, extract_last_output

class Model():

	def __init__(self, inputs, latent_mean, latent_std, is_training=True):

		self.is_training = is_training
		self.inputs = inputs
		self.n_smpl_params = cfg.CONST.nz
		self.latent_mean = latent_mean
		self.latent_std = latent_std
		self.outputs = {}
		self.use_gt_trans_params = True
		self.apply_svd_to_rot_mats = cfg.CONST.use_svd

		caption_embedding = text_encoder(self.inputs, is_training=is_training)
		self.smpl_latent = caption_embedding
		self.outputs['latent'] = self.smpl_latent
		return

	def get_outputs(self):
		return self.outputs['latent']

def text_encoder(inputs, is_training):
	embedding = inputs
	seq_length = compute_sequence_length(embedding)

	with slim.arg_scope([slim.convolution, slim.fully_connected],
						activation_fn=None,
						weights_regularizer=slim.l2_regularizer(0.005)):

		net = embedding
		rnn_cell = tf.contrib.rnn.GRUCell(num_units=10)

		net=tf.cast(net, tf.float64)
		outputs, final_state = tf.nn.dynamic_rnn(cell=rnn_cell,
												inputs=net,
												sequence_length=seq_length,
												dtype=tf.float64,
												scope='rnn')

		net = extract_last_output(outputs, seq_length)
		
		net = slim.fully_connected(net, 22, activation_fn=None, scope='fc5')
		net = slim.fully_connected(net, 32, activation_fn=None, scope='fc6')
		net = slim.fully_connected(net, 42, activation_fn=None, scope='fc7')
		net = slim.fully_connected(net, 52, activation_fn=None, scope='fc8')
		net = slim.fully_connected(net, 62, activation_fn=None, scope='fc9')
		net = slim.fully_connected(net, 72, activation_fn=None, scope='fc10')

	return net

def parameter_decoder(inputs):
	
	net = slim.stack(inputs, slim.fully_connected, [100, cfg.CONST.orig], scope='mlp')
	return net