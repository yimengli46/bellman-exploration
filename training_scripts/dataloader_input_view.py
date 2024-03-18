import numpy as np
import numpy.linalg as LA
import cv2
import matplotlib.pyplot as plt
import random
from core import cfg
import torch.utils.data as data
import torch
import torch.nn.functional as F
from random import Random
import os
import glob
import pickle
from modeling.utils.baseline_utils import apply_color_to_map
import bz2
import _pickle as cPickle

class MP3DViewDataset(data.Dataset):

	def __init__(self, split, scene_name, data_folder=''):
		self.split = split
		self.scene_name = scene_name
		
		self.saved_folder = f'{data_folder}/{self.split}/{self.scene_name}'

		self.sample_name_list = [os.path.splitext(os.path.basename(x))[0] for x in sorted(glob.glob(f'{self.saved_folder}/*.pbz2'))] 

	def __len__(self):
		return len(self.sample_name_list)

	def __getitem__(self, i):
		if self.split == 'val':
			#============================= load npy file and pickle file ===============================
			with bz2.BZ2File(f'{self.saved_folder}/{self.sample_name_list[i]}.pbz2', 'rb') as fp:
				npy_file = cPickle.load(fp)
				rgb_obs = npy_file['rgb']
				depth_obs = npy_file['depth']
				sseg_obs = npy_file['sseg']
				R = npy_file['R']
				D = npy_file['D']
				Din = npy_file['Din']
				Dout = npy_file['Dout']
		elif self.split == 'train':
			with bz2.BZ2File(f'{self.saved_folder}/{self.sample_name_list[i]}.pbz2', 'rb') as fp:
				npy_file = cPickle.load(fp)
				rgb_obs = npy_file['rgb']
				depth_obs = npy_file['depth']
				sseg_obs = npy_file['sseg']
				R = npy_file['R']
				D = npy_file['D']
				Din = npy_file['Din']
				Dout = npy_file['Dout']

		sseg_obs = np.where(sseg_obs < 0, 0, sseg_obs)

		#=================================== visualize M_p =========================================
		if cfg.PRED.VIEW.FLAG_VISUALIZE_PRED_LABELS:
			print(f'R = {R}, D = {D}, Din = {Din}, Dout = {Dout}')
			color_sseg_obs = apply_color_to_map(sseg_obs)

			fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
			ax[0].imshow(rgb_obs, cmap='gray')
			ax[0].get_xaxis().set_visible(False)
			ax[0].get_yaxis().set_visible(False)
			ax[0].set_title('input: occupancy_map_Mp')
			ax[1].imshow(color_sseg_obs)
			ax[1].get_xaxis().set_visible(False)
			ax[1].get_yaxis().set_visible(False)
			ax[1].set_title('input: semantic_map_Mp')
			ax[2].imshow(depth_obs, cmap='gray')
			ax[2].get_xaxis().set_visible(False)
			ax[2].get_yaxis().set_visible(False)
			ax[2].set_title('observed_occ_map + frontiers')

			fig.tight_layout()
			plt.show()

		#================= convert to tensor=================
		tensor_depth = torch.tensor(depth_obs, dtype=torch.float32)
		tensor_sseg = torch.tensor(sseg_obs, dtype=torch.float32)
		#print(f'tensor_depth.shape = {tensor_depth.shape}')
		#print(f'tensor_sseg.shape = {tensor_sseg.shape}')

		#================= convert input tensor into one-hot vector===========================
		tensor_depth = tensor_depth.unsqueeze(0)
		tensor_sseg = tensor_sseg.unsqueeze(0)
		#tensor_sseg = F.one_hot(tensor_sseg, num_classes=cfg.SEM_MAP.GRID_CLASS_SIZE).permute(2, 0, 1) # num_classes x H x W
		#print(f'tensor_sseg.shape = {tensor_sseg.shape}')
		tensor_input = torch.cat((tensor_depth, tensor_sseg), 0).float()
		tensor_output = torch.tensor([R, D, Din, Dout]).float()

		if cfg.PRED.VIEW.INPUT == 'depth_only':
			tensor_input = tensor_input[0].unsqueeze(0)

		#print(f'tensor_input.shape = {tensor_input.shape}')
		#print(f'tensor_output.shape = {tensor_output.shape}')

		return {'input': tensor_input, 'output': tensor_output}


def get_all_view_dataset(split, scene_list, data_folder):
	ds_list = []
	for scene in scene_list:
		scene_ds = MP3DViewDataset(split, scene, data_folder=data_folder)
		ds_list.append(scene_ds)

	concat_ds = data.ConcatDataset(ds_list)
	return concat_ds

if __name__ == "__main__":
	cfg.merge_from_file('configs/exp_train_input_view_depth_and_sem.yaml')
	cfg.freeze()

	split = 'val'
	if split == 'train':
		scene_list = cfg.MAIN.TRAIN_SCENE_LIST
	elif split == 'val':
		scene_list = cfg.MAIN.VAL_SCENE_LIST
	elif split == 'test':
		scene_list = cfg.MAIN.TEST_SCENE_LIST

	data_folder = 'output/training_data_input_view_1000samples'

	ds_list = []
	for scene in scene_list:
		scene_ds = MP3DViewDataset(split, scene, data_folder=data_folder)
		ds_list.append(scene_ds)

	concat_ds = data.ConcatDataset(ds_list)
