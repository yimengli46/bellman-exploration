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

class MP3DSceneDataset(data.Dataset):

	def __init__(self, split, scene_name, data_folder=''):
		self.split = split
		self.scene_name = scene_name
		
		self.saved_folder = f'{data_folder}/{self.split}/{self.scene_name}'

		self.sample_name_list = [os.path.splitext(os.path.basename(x))[0] for x in sorted(glob.glob(f'{self.saved_folder}/*.npy'))] 


	def __len__(self):
		return len(self.sample_name_list)

	def __getitem__(self, i):
		#============================= load npy file and pickle file ===============================
		npy_file = np.load(f'{self.saved_folder}/{self.sample_name_list[i]}.npy', allow_pickle=True).item()
		pk_file = pickle.load(open(f'{self.saved_folder}/{self.sample_name_list[i]}.pkl', 'rb'))

		M_p = npy_file['Mp']
		U_a = npy_file['Ua']
		U_d = npy_file['Ud']
		frontiers = pk_file
		#print(f'U_a.shape = {U_a[..., np.newaxis].shape}')
		#print(f'U_d.shape = {U_d.shape}')
		U_all = np.concatenate((U_a[..., np.newaxis], U_d), axis=2)

		H, W = M_p.shape[1], M_p.shape[2]
		# there are class 99 in the sem map
		M_p[1] = np.where(M_p[1] >= cfg.SEM_MAP.GRID_CLASS_SIZE, 0, M_p[1])

		#=================================== visualize M_p =========================================
		if cfg.PRED.PARTIAL_MAP.FLAG_VISUALIZE_PRED_LABELS:
			occ_map_Mp = M_p[0]
			sem_map_Mp = M_p[1]
			color_sem_map_Mp = apply_color_to_map(sem_map_Mp)

			fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
			ax[0][0].imshow(occ_map_Mp, cmap='gray')
			ax[0][0].get_xaxis().set_visible(False)
			ax[0][0].get_yaxis().set_visible(False)
			ax[0][0].set_title('input: occupancy_map_Mp')
			ax[0][1].imshow(color_sem_map_Mp)
			ax[0][1].get_xaxis().set_visible(False)
			ax[0][1].get_yaxis().set_visible(False)
			ax[0][1].set_title('input: semantic_map_Mp')

			ax[1][0].imshow(occ_map_Mp, cmap='gray')
			for f in frontiers:
				ax[1][0].scatter(f.points[1], f.points[0], c='yellow', zorder=2)
				ax[1][0].scatter(f.centroid[1], f.centroid[0], c='red', zorder=2)
			ax[1][0].get_xaxis().set_visible(False)
			ax[1][0].get_yaxis().set_visible(False)
			ax[1][0].set_title('observed_occ_map + frontiers')

			ax[1][1].imshow(U_a, vmin=0.0, vmax=1.0)
			ax[1][1].get_xaxis().set_visible(False)
			ax[1][1].get_yaxis().set_visible(False)
			ax[1][1].set_title('output: U_a')

			fig.tight_layout()
			plt.show()

		# resize M_p and U_a
		num_channels = M_p.shape[0] # 2 channels: occ and sem
		resized_Mp = np.zeros((num_channels, cfg.PRED.PARTIAL_MAP.INPUT_WH[1], cfg.PRED.PARTIAL_MAP.INPUT_WH[0]), dtype=np.float32)
		resized_Mp[0] = cv2.resize(M_p[0], cfg.PRED.PARTIAL_MAP.INPUT_WH, interpolation=cv2.INTER_NEAREST)
		resized_Mp[1] = cv2.resize(M_p[1], cfg.PRED.PARTIAL_MAP.INPUT_WH, interpolation=cv2.INTER_NEAREST)

		resized_U = np.zeros((4, cfg.PRED.PARTIAL_MAP.INPUT_WH[1], cfg.PRED.PARTIAL_MAP.INPUT_WH[0]), dtype=np.float32)
		resized_U[0] = cv2.resize(U_a, cfg.PRED.PARTIAL_MAP.INPUT_WH, interpolation=cv2.INTER_NEAREST)
		resized_U[1] = cv2.resize(U_d[:,:,0], cfg.PRED.PARTIAL_MAP.INPUT_WH, interpolation=cv2.INTER_NEAREST)
		resized_U[2] = cv2.resize(U_d[:,:,1], cfg.PRED.PARTIAL_MAP.INPUT_WH, interpolation=cv2.INTER_NEAREST)
		resized_U[3] = cv2.resize(U_d[:,:,2], cfg.PRED.PARTIAL_MAP.INPUT_WH, interpolation=cv2.INTER_NEAREST)

		#================= convert to tensor=================
		tensor_Mp = torch.tensor(resized_Mp, dtype=torch.long)
		tensor_U = torch.tensor(resized_U, dtype=torch.float32)

		#print(f'tensor_Mp.max = {torch.max(tensor_Mp)}')
		#================= convert input tensor into one-hot vector===========================
		tensor_Mp_occ = tensor_Mp[0] # H x W
		tensor_Mp_occ = F.one_hot(tensor_Mp_occ, num_classes=3).permute(2, 0, 1) # 3 x H x W
		tensor_Mp_sem = tensor_Mp[1]
		tensor_Mp_sem = F.one_hot(tensor_Mp_sem, num_classes=cfg.SEM_MAP.GRID_CLASS_SIZE).permute(2, 0, 1) # num_classes x H x W
		tensor_Mp = torch.cat((tensor_Mp_occ, tensor_Mp_sem), 0).float()

		if cfg.PRED.PARTIAL_MAP.INPUT == 'occ_only':
			tensor_Mp = tensor_Mp[0:3]

		return {'input': tensor_Mp, 'output': tensor_U, 'shape': (H, W), 'frontiers': frontiers,
			'original_target': U_all}


def get_all_scene_dataset(split, scene_list, data_folder):
	ds_list = []
	for scene in scene_list:
		scene_ds = MP3DSceneDataset(split, scene, data_folder=data_folder)
		ds_list.append(scene_ds)

	concat_ds = data.ConcatDataset(ds_list)
	return concat_ds

def my_collate(batch):
	output_dict = {}
	#==================================== for input ==================================
	out = None
	batch_input = [dict['input'] for dict in batch]
	output_dict['input'] = torch.stack(batch_input, 0)

	#==================================== for output ==================================
	out = None
	batch_output = [dict['output'] for dict in batch]
	output_dict['output'] = torch.stack(batch_output, 0)

	batch_shape = [dict['shape'] for dict in batch]
	output_dict['shape'] = batch_shape

	batch_frontiers = [dict['frontiers'] for dict in batch]
	output_dict['frontiers'] = batch_frontiers

	batch_target = [dict['original_target'] for dict in batch]
	output_dict['original_target'] = batch_target

	return output_dict

if __name__ == "__main__":
	cfg.merge_from_file('configs/exp_train_input_partial_map.yaml')
	cfg.freeze()

	split = 'train'
	if split == 'train':
		scene_list = cfg.MAIN.TRAIN_SCENE_LIST
	elif split == 'val':
		scene_list = cfg.MAIN.VAL_SCENE_LIST
	elif split == 'test':
		scene_list = cfg.MAIN.TEST_SCENE_LIST
	
	data_folder = 'output/training_data_input_partial_map'

	ds_list = []
	for scene in scene_list:
		scene_ds = MP3DSceneDataset(split, scene, data_folder=data_folder)
		ds_list.append(scene_ds)

	concat_ds = data.ConcatDataset(ds_list)
