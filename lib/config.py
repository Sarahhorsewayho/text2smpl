import os

from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.CONST = edict()
__C.CONST.orig = 72
__C.CONST.orig_nz = 72
__C.CONST.nz = 72
__C.CONST.use_svd = True
__C.CONST.batch_size = 64
__C.CONST.shuffle_size = 10
__C.CONST.weight_decay = 0.0001
__C.CONST.lr = 0.04
__C.CONST.beta1 = 0.5
__C.CONST.max_epochs = 300

__C.CONST.mode = 'train'

__C.DIR = edict()
__C.DIR.data_path = '/home/harryh/t2b/data'
__C.DIR.body_path = '/home/harryh/t2b/data/up-3d'
__C.DIR.train_caption_path = '/home/harryh/t2b/data/captions/train_captions.csv'
__C.DIR.test_caption_path = '/home/harryh/t2b/data/captions/test_captions.csv'
__C.DIR.word2vec_model_path = '/home/harryh/t2b/data/MyModel'

__C.DIR.smplparams_mean = '/home/harryh/t2b/data/helper_data/stats/train_mean.npy'
__C.DIR.smplparams_std = '/home/harryh/t2b/data/helper_data/stats/train_std.npy'

__C.DIR.checkpoint_path = '/home/harryh/t2b/data/Model'