import numpy as np
import numpy.linalg as LA
import cv2
import matplotlib.pyplot as plt
import random
from modeling.utils.baseline_utils import apply_color_to_map, pose_to_coords, gen_arrow_head_marker, read_map_npy, read_occ_map_npy, plus_theta_fn
from core import cfg
import modeling.utils.frontier_utils as fr_utils
from modeling.localNavigator_Astar import localNav_Astar
import networkx as nx
import torch.utils.data as data
import torch
import torch.nn.functional as F
from random import Random
from timeit import default_timer as timer
from itertools import islice

SEED = cfg.GENERAL.RANDOM_SEED
random.seed(SEED)
np.random.seed(SEED)

def get_region(robot_pos, H, W, size=2):
	y, x = robot_pos
	y1 = max(0, y-size)
	y2 = min(H-1, y+size)
	x1 = max(0, x-size)
	x2 = min(W-1, x+size)

	return (y1, x1, y2, x2)

class MP3DIterator:
	"""Class to implement an iterator
	of powers of two"""

	def __init__(self, split, scene_names, num_elems=0, seed=0):
		self.num_elems = num_elems
		self.random = Random(seed)

		self.split = split
		self.scene_names = scene_names
		
		self.init_scene()
		
	def init_scene(self):
		scene_name = self.random.choice(self.scene_names)
		print(f'init new scene: {scene_name}')

		#======================================== generate gt map M_c ============================================
		sem_map_npy = np.load(f'{cfg.SAVE.SEMANTIC_MAP_PATH}/{self.split}/{scene_name}/BEV_semantic_map.npy', allow_pickle=True).item()
		self.gt_sem_map, self.pose_range, self.coords_range, self.WH = read_map_npy(sem_map_npy)
		occ_map_npy = np.load(f'{cfg.SAVE.OCCUPANCY_MAP_PATH}/{self.split}/{scene_name}/BEV_occupancy_map.npy', allow_pickle=True).item()
		gt_occ_map, _, _, _ = read_occ_map_npy(occ_map_npy)

		gt_occupancy_map = gt_occ_map.copy()
		gt_occupancy_map = np.where(gt_occupancy_map == 1, cfg.FE.FREE_VAL, gt_occupancy_map)  # free cell
		self.gt_occupancy_map = np.where(gt_occupancy_map == 0, cfg.FE.COLLISION_VAL, gt_occupancy_map)  # occupied cell

		self.M_c = np.stack((self.gt_occupancy_map, self.gt_sem_map))
		self.H, self.W = self.gt_sem_map.shape

		self.LN = localNav_Astar(self.pose_range, self.coords_range, self.WH, scene_name)

		self.G = self.LN.get_G_from_map(gt_occupancy_map)
		self.largest_cc = list(max(nx.connected_components(self.G), key=len))
		
		self.i_loc = 0
		self.path = None

	def get_next(self):
		if self.path is None or self.i_loc == len(self.path):

			if self.random.random() > cfg.PRED.RENEW_SCENE_THRESH:
				self.init_scene()

			#====================================== generate (start, goal) locs, compute path P==========================
			start_loc, goal_loc = self.random.choices(self.largest_cc, k=2)
			self.path = nx.shortest_path(self.G,
									source=start_loc,
									target=goal_loc)

			self.M_p = np.zeros(self.M_c.shape, dtype=np.int16)
			self.observed_area_flag = np.zeros((self.H, self.W), dtype=bool)
			self.i_loc = 0

		robot_loc = self.path[self.i_loc]

		#=================================== generate partial map M_p ==================================
		roi = get_region(robot_loc, self.H, self.W, size=cfg.PRED.NEIGHBOR_SIZE)
		self.M_p[:, roi[0]:roi[2]+1, roi[1]:roi[3]+1] = self.M_c[:, roi[0]:roi[2]+1, roi[1]:roi[3]+1]
		self.observed_area_flag[roi[0]:roi[2]+1, roi[1]:roi[3]+1] = True

		#================================= compute area at frontier points ========================
		U_a = np.zeros((self.H, self.W), dtype=np.float32)
		observed_occupancy_map = self.M_p[0]
		frontiers = fr_utils.get_frontiers(observed_occupancy_map, self.gt_occupancy_map, self.observed_area_flag)
		agent_map_pose = (robot_loc[1], robot_loc[0])
		frontiers = self.LN.filter_unreachable_frontiers_temp(frontiers, agent_map_pose, observed_occupancy_map)

		for fron in frontiers:
			points = fron.points.transpose() # N x 2
			R = min(1. * fron.R / cfg.PRED.MAX_AREA, 1.0)
			U_a[points[:, 0], points[:, 1]] = R

		#=================================== visualize M_p =========================================
		if cfg.PRED.FLAG_VISUALIZE_PRED_LABELS:
			occ_map_Mp = self.M_p[0]
			sem_map_Mp = self.M_p[1]
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
			x_coord_lst = [self.path[i][1] for i in range(i_loc+1)]
			z_coord_lst = [self.path[i][0] for i in range(i_loc+1)]
			ax[1][0].plot(x_coord_lst, z_coord_lst, lw=5, c='blue', zorder=3)
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

		self.i_loc += 1

		# resize M_p and U_a
		num_channels = self.M_p.shape[0]
		resized_Mp = np.zeros((num_channels, cfg.PRED.INPUT_WH[1], cfg.PRED.INPUT_WH[0]), dtype=np.float32)
		resized_Mp[0] = cv2.resize(self.M_p[0], cfg.PRED.INPUT_WH, interpolation=cv2.INTER_NEAREST)
		resized_Mp[1] = cv2.resize(self.M_p[1], cfg.PRED.INPUT_WH, interpolation=cv2.INTER_NEAREST)

		resized_Ua = cv2.resize(U_a, cfg.PRED.INPUT_WH, interpolation=cv2.INTER_NEAREST)

		#================= convert to tensor=================
		tensor_Mp = torch.tensor(resized_Mp)
		tensor_Ua = torch.tensor(resized_Ua).unsqueeze(0)
		return {'input': tensor_Mp, 'output': tensor_Ua, 'shape': (self.H, self.W), 'frontiers': frontiers, \
			'original_target': U_a}

	def __iter__(self):
		self.n = 0
		return self

	def __next__(self):
		if self.n < self.num_elems:
			result = self.get_next()
			self.n += 1
			return result
		else:
			raise StopIteration

class MP3DDataset(data.IterableDataset):
	def __init__(self, split='train', scene_names=[], worker_size=0, seed=0, num_elems=10000):
		super(MP3DDataset).__init__()
		self.worker_size = worker_size
		random = Random(seed)
		self.seeds = random.sample(range(0, 100), max(self.worker_size, 1))
		self.split = split
		self.scene_names = scene_names
		self.num_elems = num_elems

		self.streams = self.get_streams()
		
	def get_stream(self, seed=0):
		#scene_name = self.scene_names[0]
		randIter = MP3DIterator(split=self.split, scene_names=self.scene_names, num_elems=self.num_elems, seed=seed)
		return iter(randIter) #map(self.process_data, iter(randIter))

	def get_streams(self):
		lst_streams = []
		for i in range(max(self.worker_size, 1)):
			lst_streams.append(self.get_stream(seed=self.seeds[i]))
		return lst_streams

	def __iter__(self):
		#return iter(self.streams[0])
		worker_info = data.get_worker_info()
		if worker_info is None:
			return iter(self.streams[0])
		else:
			worker_id = worker_info.id 
			print(f'worker_id = {worker_id}')
			return iter(self.streams[worker_id])

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


'''
device = torch.device('cuda')

env_scene = '17DRP5sb8fy'
floor_id = 0
scene_name = '17DRP5sb8fy_0'

ds = MP3DDataset(scene_names=[scene_name], worker_size=cfg.PRED.NUM_WORKERS, seed=cfg.GENERAL.RANDOM_SEED)
loader = data.DataLoader(ds, batch_size=cfg.PRED.BATCH_SIZE, num_workers=cfg.PRED.NUM_WORKERS)

t0 = timer()
i = 0
for batch in islice(loader, 50):
	print(f'i = {i}')
	i += 1
t1 = timer()
print(f'build map time = {t1 - t0}')
'''