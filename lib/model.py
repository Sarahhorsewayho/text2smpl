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
		with tf.variable_scope("generator") as sf:
			self.smpl_latent = parameter_decoder(caption_embedding)

		#smpl_shape_pred, smpl_pose_pred = \
		#	self.postprocess_smpl_params(10, 216, 24)
		self.outputs['latent'] = self.smpl_latent
		return

	#def postprocess_smpl_params(self, n_shape, n_pose, n_joints):

	#	smpl_shape_pred = self.smpl_latent[:,:n_shape] + self.latent_mean[:n_shape]
	#	smpl_pose_pred = tf.reshape(self.smpl_latent[:,n_shape:n_shape+n_pose] + self.latent_mean[n_shape:n_shape+n_pose], [-1, n_joints, 3, 3])

	#	if self.apply_svd_to_rot_mats:
	#		w, u, v = tf.svd(smpl_pose_pred, full_matrices=True)
	#		smpl_pose_pred = tf.matmul(u, tf.transpose(v, perm=[0,1,3,2]))

	#	return smpl_shape_pred, smpl_pose_pred

	def get_outputs(self):
		return self.outputs['latent']


def text_encoder(inputs, is_training):
	embedding = inputs
	print("look embedding:")
	print(embedding)
	length = cfg.CONST.batch_size

	with slim.arg_scope([slim.convolution, slim.fully_connected],
						activation_fn=tf.nn.relu,
						weights_regularizer=slim.l2_regularizer(0.005)):
	
		net = slim.convolution(embedding, 128, 3, scope='conv1')
		net = slim.convolution(net, 128, 3, scope='conv2')
		net = tf.layers.batch_normalization(net, training=is_training)
		net = slim.convolution(net, 256, 3, scope='conv3')
		net = slim.convolution(net, 256, 3, scope='conv4') 

		net = tf.layers.batch_normalization(net, training=is_training)

		rnn_cell = tf.contrib.rnn.GRUCell(num_units=256)

		net=tf.cast(net, tf.float64)
		outputs, final_state = tf.nn.dynamic_rnn(cell=rnn_cell,
												inputs=net,
												dtype=tf.float64,
												scope='rnn')

		net = extract_last_output(outputs)
		net = slim.fully_connected(net, 256, scope='fc5')
		net = slim.fully_connected(net, 128, activation_fn=None, scope='fc6')

	return net

def parameter_decoder(inputs):
	
	net = slim.stack(inputs, slim.fully_connected, [100, 72], scope='mlp')
	#net = tf.cast(net, tf.float64, name="final_out")
	#print("look net")
	#print(net)
	return net