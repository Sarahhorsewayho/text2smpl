import numpy as np 
import os
import pickle
from lib.config import cfg
from lib.write_mesh import write
from copy import copy as _copy
from smpl_webuser.serialization import load_model
from up_tools.mesh import Mesh as _Mesh
from up_tools.camera import (rotateY as _rotateY)
from up_tools.bake_vertex_colors import bake_vertex_colors
from lib.conversions import rotmat_to_aar, prepare_kintree

MODEL_NEUTRAL_PATH = '/home/harryh/t2b/data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
MODEL_NEUTRAL = load_model(MODEL_NEUTRAL_PATH)

_TEMPLATE_MESH = _Mesh(filename='/home/harryh/t2b/data/template-bodyparts.ply')


def write_to_mesh(predicted, latent_mean, latent_std):

	#kintree = prepare_kintree()
	#shape_params = np.zeros((cfg.CONST.batch_size, 10))
	#pose_params = np.zeros((cfg.CONST.batch_size, 72))
	#predicted = predicted + latent_mean
	#for i in range(cfg.CONST.batch_size):
		#params = predicted[i]
		#shape_param = params[0:10]
		#pose_param = params[10:226]
		#pose_param = rotmat_to_aar(pose_param, kintree)
		#shape_params[i] = shape_param
		#pose_params[i] = pose_param
	for i in range (cfg.CONST.batch_size):
		mesh = _copy(_TEMPLATE_MESH)

		model = MODEL_NEUTRAL
	#model.betas[:len(shape_params[0])] = shape_params[0]
		model.betas[:10] = np.zeros((10,))
		model.pose[:] = predicted[i]
		#print(predicted[0])
		model.trans[:] = [-0.01, 0.115, 20.3]
	#f = open('/home/harryh/t2b/data/up-3d/00001_body.pkl', 'rb')
	#info = pickle.load(f, encoding="bytes")
	#model.betas[:10] = np.zeros((10,))
	#model.pose[:] = info[b'pose']

	
    #model.trans[:] = [-0.01, 0.115, 20.3]
	
	#latent_pose = rotmat_to_aar(latent_mean[10:226], kintree)
	#model.betas[:10] = latent_mean[:10]
	#model.pose[:] = latent_pose
	#model.trans[:] = [-0.01, 0.115, 20.3]

		mesh.v = model.r
		baked_mesh = bake_vertex_colors(mesh)
		base_mesh = _copy(baked_mesh)
		mesh.f = base_mesh.f
		mesh.vc = base_mesh.vc

	#mean = np.mean(np.array(base_mesh.v), axis=0, keepdims=True)
		mesh.v = np.array(base_mesh.v)
	#mean = np.mean(np.array(base_mesh.v), axis=0, keepdims=True)
	#mesh.v = _rotateY((np.array(base_mesh.v)-mean),angle) + mean

		path = '/home/harryh/t2b/data/' + str(i) + '.obj'
	#if not os.path.exists(path):
	#	os.mkdir(path)

		write(path, mesh.v, mesh.f)

