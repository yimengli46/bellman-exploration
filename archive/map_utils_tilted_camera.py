import numpy as np
import numpy.linalg as LA
import cv2
import matplotlib.pyplot as plt
import math
from math import cos, sin, acos, atan2, pi, floor
from .baseline_utils import project_pixels_to_world_coords, convertPanopSegToSSeg, apply_color_to_map, pose_to_coords, convertInsSegToSSeg
from .baseline_utils import pxl_coords_to_pose
from core import cfg
from .build_map_utils import find_first_nonzero_elem_per_row
from timeit import default_timer as timer

""" class used to build semantic maps of the scenes

The robot takes actions in the environment and use the observations to build the semantic map online.
"""


class SemanticMap:

	def __init__(self, split, scene_name, pose_range, coords_range, WH, ins2cat_dict):
		self.split = split
		self.scene_name = scene_name
		self.cell_size = cfg.SEM_MAP.CELL_SIZE
		self.detector = cfg.NAVI.DETECTOR
		#self.panop_pred = PanopPred()
		self.pose_range = pose_range
		self.coords_range = coords_range
		self.WH = WH
		self.occupied_poses = []  # detected during navigation

		self.IGNORED_CLASS = cfg.SEM_MAP.IGNORED_SEM_CLASS  # ceiling class is ignored
		self.UNDETECTED_PIXELS_CLASS = cfg.SEM_MAP.UNDETECTED_PIXELS_CLASS

		self.ins2cat_dict = ins2cat_dict

		# load occupancy map
		occ_map_path = f'{cfg.SAVE.OCCUPANCY_MAP_PATH}/{self.split}/{self.scene_name}'
		self.occupancy_map = np.load(f'{occ_map_path}/BEV_occupancy_map.npy',
									 allow_pickle=True).item()['occupancy']
		#kernel = np.ones((5,5), np.uint8)
		#self.occupancy_map = cv2.erode(occupancy_map.astype(np.uint8), kernel, iterations=1)
		print(f'self.occupancy_map.shape = {self.occupancy_map.shape}')

		# ==================================== initialize 4d grid =================================
		self.min_X = -cfg.SEM_MAP.WORLD_SIZE
		self.max_X = cfg.SEM_MAP.WORLD_SIZE
		self.min_Z = -cfg.SEM_MAP.WORLD_SIZE
		self.max_Z = cfg.SEM_MAP.WORLD_SIZE

		self.x_grid = np.arange(self.min_X, self.max_X, self.cell_size)
		self.z_grid = np.arange(self.min_Z, self.max_Z, self.cell_size)

		self.four_dim_grid = np.zeros(
			(len(self.z_grid), cfg.SEM_MAP.GRID_Y_SIZE, len(
				self.x_grid), cfg.SEM_MAP.GRID_CLASS_SIZE),
			dtype=np.int16)  # x, y, z, C
		self.H, self.W = len(self.z_grid), len(self.x_grid)

	def build_semantic_map(self, obs_list, pose_list, step=0, saved_folder=''):
		""" update semantic map with observations rgb_img, depth_img, sseg_img and robot pose."""
		assert len(obs_list) == len(pose_list)
		for idx, obs in enumerate(obs_list):
			pose = pose_list[idx]
			# load rgb image, depth and sseg
			rgb_img = obs['rgb']
			depth_img = 5. * obs['depth']
			depth_img = cv2.blur(depth_img, (3, 3))
			#print(f'depth_img.shape = {depth_img.shape}')
			InsSeg_img = obs["semantic"]
			sseg_img = convertInsSegToSSeg(InsSeg_img, self.ins2cat_dict)
			sem_map_pose = (pose[0], -pose[1], -pose[2])  # x, z, theta
			#print('pose = {}'.format(pose))

			#'''
			#if step % 10 == 0:
			if cfg.SEM_MAP.FLAG_VISUALIZE_EGO_OBS:
				fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))
				ax[0].imshow(rgb_img)
				ax[0].get_xaxis().set_visible(False)
				ax[0].get_yaxis().set_visible(False)
				ax[0].set_title("rgb")
				ax[1].imshow(apply_color_to_map(sseg_img))
				ax[1].get_xaxis().set_visible(False)
				ax[1].get_yaxis().set_visible(False)
				ax[1].set_title("sseg")
				ax[2].imshow(depth_img)
				ax[2].get_xaxis().set_visible(False)
				ax[2].get_yaxis().set_visible(False)
				ax[2].set_title("depth")
				fig.tight_layout()
				plt.show()
				#fig.savefig(f'{saved_folder}/step_{step}_obs.jpg')
				#plt.close()
			#'''
			if cfg.NAVI.HFOV == 90:
				xyz_points, sseg_points = project_pixels_to_world_coords(
					sseg_img,
					depth_img,
					sem_map_pose,
					gap=2,
					FOV=90,
					cx=160,
					cy=320,
					resolution_x=320,
					resolution_y=640,
					theta_x=-0.785,
					ignored_classes=self.IGNORED_CLASS)
			elif cfg.NAVI.HFOV == 360:
				xyz_points, sseg_points = project_pixels_to_world_coords(
					sseg_img,
					depth_img,
					sem_map_pose,
					gap=2,
					FOV=120,
					cx=240,
					cy=320,
					resolution_x=480,
					resolution_y=640,
					theta_x=-0.785,
					ignored_classes=self.IGNORED_CLASS)

			#print(f'xyz_points.shape = {xyz_points.shape}')
			#print(f'sseg_points.shape = {sseg_points.shape}')

			mask_X = np.logical_and(xyz_points[0, :] > self.min_X,
									xyz_points[0, :] < self.max_X)
			mask_Y = np.logical_and(xyz_points[1, :] > 0.0,
									xyz_points[1, :] < 100.0)
			mask_Z = np.logical_and(xyz_points[2, :] > self.min_Z,
									xyz_points[2, :] < self.max_Z)
			mask_XYZ = np.logical_and.reduce((mask_X, mask_Y, mask_Z))
			xyz_points = xyz_points[:, mask_XYZ]
			sseg_points = sseg_points[mask_XYZ]

			x_coord = np.floor(
				(xyz_points[0, :] - self.min_X) / self.cell_size).astype(int)
			y_coord = np.floor(xyz_points[1, :] / self.cell_size).astype(int)
			z_coord = (self.H - 1) - np.floor(
				(xyz_points[2, :] - self.min_Z) / self.cell_size).astype(int)
			mask_y_coord = y_coord < cfg.SEM_MAP.GRID_Y_SIZE
			x_coord = x_coord[mask_y_coord]
			y_coord = y_coord[mask_y_coord]
			z_coord = z_coord[mask_y_coord]
			sseg_points = sseg_points[mask_y_coord]
			self.four_dim_grid[z_coord, y_coord, x_coord, sseg_points] += 1
		#assert 1==2

	def get_semantic_map(self):
		""" get the built semantic map. """
		# reduce size of the four_dim_grid
		smaller_four_dim_grid = self.four_dim_grid[self.coords_range[1]:self.coords_range[3] + 1, 
													:, self.coords_range[0]:self.coords_range[2] + 1, :] 
		# argmax over the category axis
		zyx_grid = np.argmax(smaller_four_dim_grid, axis=3)
		# swap y dim to the last axis
		zxy_grid = np.swapaxes(zyx_grid, 1, 2)
		L, M, N = zxy_grid.shape
		zxy_grid = zxy_grid.reshape(L * M, N)

		semantic_map = find_first_nonzero_elem_per_row(zxy_grid)
		semantic_map = semantic_map.reshape(L, M)

		# sum over the height axis
		grid_sum_height = np.sum(smaller_four_dim_grid, axis=1)

		grid_sum_cat = np.sum(grid_sum_height, axis=2)
		observed_area_flag = (grid_sum_cat > 0)
		#observed_area_flag = (observed_area_flag[self.coords_range[1]:self.coords_range[3]+1, self.coords_range[0]:self.coords_range[2]+1])

		# get occupancy map
		'''
		occupancy_map = np.zeros(semantic_map.shape, dtype=np.int8)
		occupancy_map = np.where(semantic_map==57, 3, occupancy_map) # floor index 57, free space index 3
		occupancy_map = np.where(semantic_map==self.UNDETECTED_PIXELS_CLASS, 2, occupancy_map) # explored but undetected area, index 2
		# occupied area are the explored area but not floor
		mask_explored_occupied_area = np.logical_and(observed_area_flag, occupancy_map==0)
		occupancy_map[mask_explored_occupied_area] = 1 # occupied space index

		occupancy_map = occupancy_map[self.coords_range[1]:self.coords_range[3]+1, self.coords_range[0]:self.coords_range[2]+1]
		'''

		occupancy_map = self.occupancy_map.copy()
		occupancy_map = np.where(occupancy_map == 1, cfg.FE.FREE_VAL,
								 occupancy_map)  # free cell
		occupancy_map = np.where(occupancy_map == 0, cfg.FE.COLLISION_VAL,
								 occupancy_map)  # occupied cell

		# add occupied cells
		for pose in self.occupied_poses:
			coords = pose_to_coords(pose,
									self.pose_range,
									self.coords_range,
									self.WH,
									flag_cropped=True)
			print(f'occupied cell coords = {coords}')
			occupancy_map[coords[1], coords[0]] = cfg.FE.COLLISION_VAL
		'''
		temp_semantic_map = semantic_map[self.coords_range[1]:self.coords_range[3]+1, self.coords_range[0]:self.coords_range[2]+1]
		temp_occupancy_map = occupancy_map[self.coords_range[1]:self.coords_range[3]+1, self.coords_range[0]:self.coords_range[2]+1]
		fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(200, 100))
		# visualize gt semantic map
		ax[0].imshow(temp_semantic_map)
		ax[0].get_xaxis().set_visible(False)
		ax[0].get_yaxis().set_visible(False)
		ax[0].set_title('semantic map')
		ax[1].imshow(temp_occupancy_map, vmax=3)
		ax[1].get_xaxis().set_visible(False)
		ax[1].get_yaxis().set_visible(False)
		ax[1].set_title('occupancy map')
		plt.show()
		'''

		return semantic_map, observed_area_flag, occupancy_map

	def get_observed_occupancy_map(self):
		""" get currently maintained occupancy map """
		# reduce size of the four_dim_grid
		smaller_four_dim_grid = self.four_dim_grid[self.coords_range[1]:self.coords_range[3] + 1, 
													:, self.coords_range[0]:self.coords_range[2] + 1, :] 
		# argmax over the category axis
		zyx_grid = np.argmax(smaller_four_dim_grid, axis=3)
		# swap y dim to the last axis
		zxy_grid = np.swapaxes(zyx_grid, 1, 2)
		L, M, N = zxy_grid.shape
		zxy_grid = zxy_grid.reshape(L * M, N)

		semantic_map = find_first_nonzero_elem_per_row(zxy_grid)
		semantic_map = semantic_map.reshape(L, M)

		# sum over the height axis
		grid_sum_height = np.sum(smaller_four_dim_grid, axis=1)

		grid_sum_cat = np.sum(grid_sum_height, axis=2)
		observed_area_flag = (grid_sum_cat > 0)
		#observed_area_flag = (observed_area_flag[self.coords_range[1]:self.coords_range[3]+1, self.coords_range[0]:self.coords_range[2]+1])

		occupancy_map = self.occupancy_map.copy()
		occupancy_map = np.where(self.occupancy_map == 1, cfg.FE.FREE_VAL,
								 occupancy_map)  # free cell
		occupancy_map = np.where(self.occupancy_map == 0, cfg.FE.COLLISION_VAL,
								 occupancy_map)  # occupied cell

		# add occupied cells
		for pose in self.occupied_poses:
			coords = pose_to_coords(pose,
									self.pose_range,
									self.coords_range,
									self.WH,
									flag_cropped=True)
			print(f'occupied cell coords = {coords}')
			occupancy_map[coords[1], coords[0]] = 1

		gt_occupancy_map = occupancy_map.copy()

		# add unobserved/unexplored area
		occupancy_map[np.logical_not(
			observed_area_flag)] = cfg.FE.UNOBSERVED_VAL

		return occupancy_map, gt_occupancy_map, observed_area_flag, semantic_map

	def add_occupied_cell_pose(self, pose):
		""" get which cells are marked as occupied by the robot during navigation."""
		agent_map_pose = (pose[0], -pose[1], -pose[2])
		self.occupied_poses.append(agent_map_pose)
