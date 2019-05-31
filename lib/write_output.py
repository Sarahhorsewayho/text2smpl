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

MODEL_NEUTRAL_PATH = '/home/harryh/t2b/data/body_template/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
MODEL_NEUTRAL = load_model(MODEL_NEUTRAL_PATH)

_TEMPLATE_MESH = _Mesh(filename='/home/harryh/t2b/data/body_template/template-bodyparts.ply')


def write_to_mesh(predicted, latent_mean, latent_std):

	kintree = prepare_kintree()
	#predicted = predicted + latent_mean
	#for i in range(cfg.CONST.batch_size):
		#params = predicted[i]
		#shape_param = params[0:10]
		#pose_param = params[10:226]
		#pose_param = rotmat_to_aar(pose_param, kintree)
		#shape_params[i] = shape_param
		#pose_params[i] = pose_param
	print(predicted[0])
	#print(predicted[1])
	for i in range (cfg.CONST.batch_size):
		mesh = _copy(_TEMPLATE_MESH)
		model = MODEL_NEUTRAL
		model.betas[:10] = np.zeros((10,))
		#pose_param = rotmat_to_aar(predicted[i], kintree)
		#model.pose[:] = pose_param
		model.pose[:] = predicted[i]
		#print(predicted[0])
		model.trans[:] = [-0.01, 0.115, 20.3]

		mesh.v = model.r
		baked_mesh = bake_vertex_colors(mesh)
		base_mesh = _copy(baked_mesh)
		mesh.f = base_mesh.f
		mesh.vc = base_mesh.vc
		mesh.v = np.array(base_mesh.v)

		path = '/home/harryh/t2b/data/objs/' + str(i) + '.obj'
		write(path, mesh.v, mesh.f)


