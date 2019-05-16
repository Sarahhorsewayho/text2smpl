import numpy as np
import tensorflow as tf

from lib.config import cfg

class Optimiser():

    def __init__(self, model, targets, latent_mean, latent_std):

        self.model = model
        self.targets = targets
        self.predictions = self.model.get_outputs()
        self.latent_mean = latent_mean
        self.latent_std = latent_std
        
        self.loss_full = None
        self.losses = {}
        self.prepare_loss_ops()

        #self.learning_rate = tf.Variable(0, trainable=False)
        self.learning_rate = cfg.CONST.lr
        self.global_step = None
        self.max_steps = None
        self.train_op  = None
        self.optimisers = []
        
        return

    def prepare_train_op(self, global_step, max_steps):
        assert self.train_op is None, 'train_op already initialised!'
        self.global_step = global_step
        self.max_steps = max_steps
        self.create_optimiser()
        return self.train_op
    
    def get_loss_op(self):
        return self.loss_full

    def get_loss_dict(self):
        return self.losses                

    def prepare_loss_ops(self):
        
        z = self.predictions['latent']
        smplparams = self.targets
        
        with tf.name_scope("generator_loss"):
            # prepare placeholders for losses
            gen_loss = tf.constant(0, tf.float64)
            gen_loss_latent = tf.constant(0, tf.float64)
            gen_loss_weights = tf.constant(0, tf.float64)
            sqerr_latent = tf.zeros_like(z)
            learning_rate = tf.constant(0, tf.float64)

            mabserr  = lambda pred, gt: tf.abs(pred-gt)

            gen_loss_latent = tf.reduce_mean(mabserr(z, smplparams))
            gen_loss += 1.0 * gen_loss_latent

        self.losses['gen_loss_latent'] = gen_loss_latent

    def create_optimiser(self):

        #incr_global_step = tf.assign(self.global_step, self.global_step+1)
        #self.learning_rate = tf.train.polynomial_decay(cfg.CONST.lr, self.global_step, self.max_steps, 
        #                                                  end_learning_rate=0, power=0.9, cycle=False)
        self.optimisers.append(tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                                  beta1=cfg.CONST.beta1))
        self.train_op = self.optimisers[0].minimize(self.loss_full)   
        #self.train_op = tf.group(incr_global_step, train)
        return

    def get_learning_rate(self):
        return self.learning_rate

    def get_losses(self):
        return self.losses