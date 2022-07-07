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

def find_neighborhood(agent_coords, occupancy_map):
	radius = int(cfg.SENSOR.SENSOR_HEIGHT/cfg.SEM_MAP.CELL_SIZE)
	H, W = occupancy_map.shape
	x = np.linspace(0, W - 1, W)
	y = np.linspace(0, H - 1, H)
	xv, yv = np.meshgrid(x, y)
	cell_coords = np.stack((xv, yv), axis=2)
	cell_dist = np.sum((cell_coords - agent_coords)**2, axis=2)
	cell_mask = (cell_dist <= radius**2)
	#plt.imshow(cell_mask)
	#plt.show()
	return cell_mask

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

		#===================================== load gt occupancy map =============================
		# load occupancy map
		occ_map_path = f'output/semantic_map_temp/{self.split}/{self.scene_name}'
		gt_occupancy_map = np.load(f'{occ_map_path}/BEV_occupancy_map.npy',
		                                                         allow_pickle=True).item()['occupancy']
		gt_occupancy_map = np.where(gt_occupancy_map == 1, cfg.FE.FREE_VAL,
		                                                 gt_occupancy_map)  # free cell
		gt_occupancy_map = np.where(gt_occupancy_map == 0, cfg.FE.COLLISION_VAL,
		                                                 gt_occupancy_map)  # occupied cell
		self.gt_occupancy_map = gt_occupancy_map
		print(f'self.gt_occupancy_map.shape = {self.gt_occupancy_map.shape}')


		# ==================================== initialize 4d grid =================================
		self.min_X = -cfg.SEM_MAP.WORLD_SIZE
		self.max_X = cfg.SEM_MAP.WORLD_SIZE
		self.min_Z = -cfg.SEM_MAP.WORLD_SIZE
		self.max_Z = cfg.SEM_MAP.WORLD_SIZE
		self.min_Y = 0.0
		self.max_Y = cfg.SENSOR.AGENT_HEIGHT + self.cell_size

		self.x_grid = np.arange(self.min_X, self.max_X, self.cell_size)
		self.z_grid = np.arange(self.min_Z, self.max_Z, self.cell_size)
		self.y_grid = np.arange(self.min_Y, self.max_Y, self.cell_size)

		self.THRESHOLD_LOW = 5 # refers to index of bin height 0.2m
		self.THRESHOLD_HIGH = len(self.y_grid)

		print(f'y_grid = {self.y_grid}')
		print(f'len(y_grid) = {len(self.y_grid)}')
		print(f'thresh_low = {self.THRESHOLD_LOW}, thresh_high = {self.THRESHOLD_HIGH}')

		self.four_dim_grid = np.zeros(
			(len(self.z_grid), len(self.y_grid)+1, len(self.x_grid), cfg.SEM_MAP.GRID_CLASS_SIZE),
			dtype=np.int16)  # x, y, z, C
		print(f'self.four_dim_grid.shape = {self.four_dim_grid.shape}')

		#============================================
		self.H, self.W = len(self.z_grid), len(self.x_grid)

	def build_semantic_map(self, obs_list, pose_list, step=0, saved_folder=''):
		""" update semantic map with observations rgb_img, depth_img, sseg_img and robot pose."""
		assert len(obs_list) == len(pose_list)
		rgb_lst, depth_lst, sseg_lst = [], [], []
		for idx, obs in enumerate(obs_list):
			pose = pose_list[idx]
			# load rgb image, depth and sseg
			rgb_img = obs['rgb']
			depth_img = obs['depth'][:,:,0]
			#print(f'depth_img.shape = {depth_img.shape}')
			InsSeg_img = obs["semantic"]
			sseg_img = convertInsSegToSSeg(InsSeg_img, self.ins2cat_dict)
			sem_map_pose = (pose[0], -pose[1], -pose[2])  # x, z, theta
			#print('pose = {}'.format(pose))
			rgb_lst.append(rgb_img)
			depth_lst.append(depth_img)
			sseg_lst.append(sseg_img)

			#'''
			if cfg.SEM_MAP.FLAG_VISUALIZE_EGO_OBS and cfg.NAVI.HFOV == 90:
				fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))
				ax[0].imshow(rgb_img)
				ax[0].get_xaxis().set_visible(False)
				ax[0].get_yaxis().set_visible(False)
				ax[0].set_title("rgb")
				ax[1].imshow(apply_color_to_map(sseg_img, flag_small_categories=True))
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
					cx=128,
					cy=128,
					resolution_x=256,
					resolution_y=256,
					theta_x=0.,
					ignored_classes=self.IGNORED_CLASS)
			elif cfg.NAVI.HFOV == 360:
				xyz_points, sseg_points = project_pixels_to_world_coords(
					sseg_img,
					depth_img,
					sem_map_pose,
					gap=2,
					FOV=90,
					cx=128,
					cy=128,
					resolution_x=256,
					resolution_y=256,
					theta_x=0.,
					ignored_classes=self.IGNORED_CLASS)

			#print(f'xyz_points.shape = {xyz_points.shape}')
			#print(f'sseg_points.shape = {sseg_points.shape}')

			mask_X = np.logical_and(xyz_points[0, :] > self.min_X,
									xyz_points[0, :] < self.max_X)
			mask_Z = np.logical_and(xyz_points[2, :] > self.min_Z,
									xyz_points[2, :] < self.max_Z)
			mask_XYZ = np.logical_and.reduce((mask_X, mask_Z))
			xyz_points = xyz_points[:, mask_XYZ]
			sseg_points = sseg_points[mask_XYZ]

			x_coord = np.floor(
			(xyz_points[0, :] - self.min_X) / self.cell_size).astype(int)
			y_coord = np.digitize(xyz_points[1, :], self.y_grid)
			z_coord = (self.H - 1) - np.floor(
				(xyz_points[2, :] - self.min_Z) / self.cell_size).astype(int)

			if x_coord.shape[0] > 0:
				self.four_dim_grid[z_coord, y_coord, x_coord, sseg_points] += 1

		if cfg.SEM_MAP.FLAG_VISUALIZE_EGO_OBS and cfg.NAVI.HFOV == 360:
			panorama_rgb = np.concatenate(rgb_lst, axis=1)
			panorama_depth = np.concatenate(depth_lst, axis=1)
			panorama_sseg = np.concatenate(sseg_lst, axis=1)

			fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(15, 6))
			ax[0].imshow(panorama_rgb)
			ax[0].get_xaxis().set_visible(False)
			ax[0].get_yaxis().set_visible(False)
			ax[0].set_title("rgb")
			ax[1].imshow(apply_color_to_map(panorama_sseg))
			ax[1].get_xaxis().set_visible(False)
			ax[1].get_yaxis().set_visible(False)
			ax[1].set_title("sseg")
			ax[2].imshow(panorama_depth)
			ax[2].get_xaxis().set_visible(False)
			ax[2].get_yaxis().set_visible(False)
			ax[2].set_title("depth")
			fig.tight_layout()
			plt.show()

	def get_semantic_map(self):
		""" get the built semantic map. """
		# reduce size of the four_dim_grid
		smaller_four_dim_grid = self.four_dim_grid[self.coords_range[1]:self.coords_range[3] + 1, 
			0:self.THRESHOLD_HIGH, self.coords_range[0]:self.coords_range[2] + 1, :] 

		#======================= build semantic map ===============================
		# argmax over the category axis
		zyx_grid = np.argmax(smaller_four_dim_grid, axis=3)
		# swap y dim to the last axis
		zxy_grid = np.swapaxes(zyx_grid, 1, 2)
		L, M, N = zxy_grid.shape
		zxy_grid = zxy_grid.reshape(L * M, N)

		semantic_map = find_first_nonzero_elem_per_row(zxy_grid)
		semantic_map = semantic_map.reshape(L, M)

		#============================= build occupancy map ===================================
		# find explored region
		observed_area_flag = smaller_four_dim_grid.sum(axis=(1, 3)) > 0
		cells_in_occupied_range = smaller_four_dim_grid[:, self.THRESHOLD_LOW:self.THRESHOLD_HIGH, :].sum(axis=(1,3))
		#print(f'cells_in_occupied_range.shape = {cells_in_occupied_range.shape}')
		occupancy_map = np.zeros(observed_area_flag.shape, dtype=np.int16)
		occupancy_map[observed_area_flag == False] = cfg.FE.UNOBSERVED_VAL
		mask_occupied = np.logical_and(cells_in_occupied_range >= cfg.SEM_MAP.POINTS_CNT,
										observed_area_flag)
		mask_free = np.logical_and(mask_occupied == False,
									observed_area_flag)
		occupancy_map[mask_free] = cfg.FE.FREE_VAL
		occupancy_map[mask_occupied] = cfg.FE.COLLISION_VAL

		'''
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
		
		'''
		fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(200, 100))
		# visualize gt semantic map
		ax[0].imshow(semantic_map)
		ax[0].get_xaxis().set_visible(False)
		ax[0].get_yaxis().set_visible(False)
		ax[0].set_title('semantic map')
		ax[1].imshow(occupancy_map, cmap='gray')
		ax[1].get_xaxis().set_visible(False)
		ax[1].get_yaxis().set_visible(False)
		ax[1].set_title('occupancy map')
		plt.show()
		'''

		return semantic_map, observed_area_flag, occupancy_map

	def get_observed_occupancy_map(self, agent_map_pose):
		""" get currently maintained occupancy map """
		# reduce size of the four_dim_grid
		smaller_four_dim_grid = self.four_dim_grid[self.coords_range[1]:self.coords_range[3] + 1, 
			0:self.THRESHOLD_HIGH, self.coords_range[0]:self.coords_range[2] + 1, :] 

		#======================= build semantic map ===============================
		# argmax over the category axis
		zyx_grid = np.argmax(smaller_four_dim_grid, axis=3)
		# swap y dim to the last axis
		zxy_grid = np.swapaxes(zyx_grid, 1, 2)
		L, M, N = zxy_grid.shape
		zxy_grid = zxy_grid.reshape(L * M, N)

		semantic_map = find_first_nonzero_elem_per_row(zxy_grid)
		semantic_map = semantic_map.reshape(L, M)

		#============================= build occupancy map ===================================
		# find explored region
		observed_area_flag = smaller_four_dim_grid.sum(axis=(1, 3)) > 0
		cells_in_occupied_range = smaller_four_dim_grid[:, self.THRESHOLD_LOW:self.THRESHOLD_HIGH, :].sum(axis=(1,3))
		#print(f'cells_in_occupied_range.shape = {cells_in_occupied_range.shape}')
		occupancy_map = np.zeros(observed_area_flag.shape, dtype=np.int16)
		occupancy_map[observed_area_flag == False] = cfg.FE.UNOBSERVED_VAL
		mask_occupied = np.logical_and(cells_in_occupied_range >= cfg.SEM_MAP.POINTS_CNT,
										observed_area_flag)
		mask_free = np.logical_and(mask_occupied == False,
									observed_area_flag)
		occupancy_map[mask_free] = cfg.FE.FREE_VAL
		occupancy_map[mask_occupied] = cfg.FE.COLLISION_VAL

		'''
		fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(200, 100))
		# visualize gt semantic map
		ax[0].imshow(mask_free)
		ax[0].get_xaxis().set_visible(False)
		ax[0].get_yaxis().set_visible(False)
		ax[0].set_title('mask_free')
		ax[1].imshow(mask_occupied, cmap='gray')
		ax[1].get_xaxis().set_visible(False)
		ax[1].get_yaxis().set_visible(False)
		ax[1].set_title('mask_occupied')
		plt.show()
		'''

		#============================== complement the region near the robot ====================
		agent_coords = pose_to_coords(agent_map_pose, self.pose_range,
									  self.coords_range, self.WH)
		# find the nearby cells coordinates
		neighborhood_mask = find_neighborhood(agent_coords, occupancy_map)
		complement_mask = np.logical_and(neighborhood_mask, observed_area_flag==False)
		# change the complement area
		occupancy_map = np.where(complement_mask, cfg.FE.FREE_VAL, occupancy_map)

		'''
		# add occupied cells
		for pose in self.occupied_poses:
			coords = pose_to_coords(pose,
									self.pose_range,
									self.coords_range,
									self.WH,
									flag_cropped=True)
			print(f'occupied cell coords = {coords}')
			occupancy_map[coords[1], coords[0]] = 1
		'''

		gt_occupancy_map = self.gt_occupancy_map

		'''
		fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(200, 100))
		# visualize gt semantic map
		ax[0].imshow(apply_color_to_map(semantic_map))
		ax[0].get_xaxis().set_visible(False)
		ax[0].get_yaxis().set_visible(False)
		ax[0].set_title('semantic map')
		ax[1].imshow(occupancy_map, cmap='gray')
		ax[1].get_xaxis().set_visible(False)
		ax[1].get_yaxis().set_visible(False)
		ax[1].set_title('occupancy map')
		plt.show()
		'''

		return occupancy_map, gt_occupancy_map, observed_area_flag, semantic_map

	def add_occupied_cell_pose(self, pose):
		""" get which cells are marked as occupied by the robot during navigation."""
		agent_map_pose = (pose[0], -pose[1], -pose[2])
		self.occupied_poses.append(agent_map_pose)
