import numpy as np
import numpy.linalg as LA
import cv2
import matplotlib.pyplot as plt
import math
from math import cos, sin, acos, atan2, pi, floor
from baseline_utils import project_pixels_to_world_coords, apply_color_to_map, save_fig_through_plt

def find_first_nonzero_elem_per_row(mat):
	H, W = mat.shape
	x = np.linspace(0, W-1, W)
	y = np.linspace(0, H-1, H)
	xv, yv = np.meshgrid(x, y)

	xv[mat == 0] = W-1
	min_idx_nonzero_per_row = np.min(xv, axis=1).astype(int)
	yv = yv[:, 0].astype(int)

	result = mat[yv, min_idx_nonzero_per_row]
	return result


class SemanticMap:
	def __init__(self, saved_folder):

		self.scene_name = ''
		self.cell_size = 0.1
		self.UNIGNORED_CLASS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15, 16, 18, 19, 22, 23, 25, 27, 28, 31, 33, \
			34, 36, 37, 38, 39, 40]
		self.step_size = 100
		self.map_boundary = 5
		self.detector = None
		self.saved_folder = saved_folder

		self.IGNORED_CLASS = [] # ceiling class is ignored
		for i in range(41):
			if i not in self.UNIGNORED_CLASS:
				self.IGNORED_CLASS.append(i)

		# ==================================== initialize 4d grid =================================
		self.min_X = -50.0
		self.max_X = 50.0
		self.min_Z = -50.0
		self.max_Z = 50.0

		self.x_grid = np.arange(self.min_X, self.max_X, self.cell_size)
		self.z_grid = np.arange(self.min_Z, self.max_Z, self.cell_size)[::-1]

		self.four_dim_grid = np.zeros((len(self.z_grid), 100, len(self.x_grid), 100), dtype=np.int16) # x, y, z, C
		
		#===================================
		self.H, self.W = len(self.z_grid), len(self.x_grid)
		self.min_x_coord = self.W - 1
		self.max_x_coord = 0
		self.min_z_coord = self.H - 1
		self.max_z_coord = 0


	def build_semantic_map(self, rgb_img, depth_img, sseg_img, pose, step_):
		sem_map_pose = (pose[0], -pose[1], -pose[2]) # x, z, theta
		
		print('pose = {}'.format(pose))
		print('sem_map_pose = {}'.format(sem_map_pose))

		#'''
		if True:
			fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))
			ax[0].imshow(rgb_img)
			ax[0].get_xaxis().set_visible(False)
			ax[0].get_yaxis().set_visible(False)
			ax[0].set_title("rgb")
			ax[1].imshow(sseg_img)
			ax[1].get_xaxis().set_visible(False)
			ax[1].get_yaxis().set_visible(False)
			ax[1].set_title("sseg")
			ax[2].imshow(depth_img)
			ax[2].get_xaxis().set_visible(False)
			ax[2].get_yaxis().set_visible(False)
			ax[2].set_title("depth")
			fig.tight_layout()
			plt.show()
		#'''

		xyz_points, sseg_points = project_pixels_to_world_coords(sseg_img, depth_img, sem_map_pose, \
			gap=2, FOV=90, cx=128, cy=128, resolution_x=256, resolution_y=256, ignored_classes=self.IGNORED_CLASS)

		#xyz_points, sseg_points = project_pixels_to_world_coords(sseg_img, depth_img, sem_map_pose, gap=2, FOV=90, cx=320, cy=640, resolution_x=640, resolution_y=1280, theta_x=-0.785, ignored_classes=self.IGNORED_CLASS)

		mask_X = np.logical_and(xyz_points[0, :] > self.min_X, xyz_points[0, :] < self.max_X) 
		mask_Y = np.logical_and(xyz_points[1, :] > 0.0, xyz_points[1, :] < 100.0)
		mask_Z = np.logical_and(xyz_points[2, :] > self.min_Z, xyz_points[2, :] < self.max_Z)  
		mask_XYZ = np.logical_and.reduce((mask_X, mask_Y, mask_Z))
		xyz_points = xyz_points[:, mask_XYZ]
		sseg_points = sseg_points[mask_XYZ]

		x_coord = np.floor((xyz_points[0, :] - self.min_X) / self.cell_size).astype(int)
		y_coord = np.floor(xyz_points[1, :] / self.cell_size).astype(int)
		z_coord = (self.H-1) - np.floor((xyz_points[2, :] - self.min_Z) / self.cell_size).astype(int)
		mask_y_coord = y_coord < 1000
		x_coord = x_coord[mask_y_coord]
		y_coord = y_coord[mask_y_coord]
		z_coord = z_coord[mask_y_coord]

		if x_coord.shape[0] > 0:
			sseg_points = sseg_points[mask_y_coord]
			self.four_dim_grid[z_coord, y_coord, x_coord, sseg_points] += 1
			
			# update the weights for the local map
			self.min_x_coord = min(max(np.min(x_coord)-self.map_boundary, 0), self.min_x_coord)
			self.max_x_coord = max(min(np.max(x_coord)+self.map_boundary, self.W-1), self.max_x_coord)
			self.min_z_coord = min(max(np.min(z_coord)-self.map_boundary, 0), self.min_z_coord)
			self.max_z_coord = max(min(np.max(z_coord)+self.map_boundary, self.H-1), self.max_z_coord)

		if step_ % self.step_size == 0:
			self.get_semantic_map(step_)

	def get_semantic_map(self, step_):
		# argmax over the category axis
		zyx_grid = np.argmax(self.four_dim_grid, axis=3)
		# swap y dim to the last axis
		zxy_grid = np.swapaxes(zyx_grid, 1, 2)
		L, M, N = zxy_grid.shape
		zxy_grid = zxy_grid.reshape(L*M, N)

		semantic_map = find_first_nonzero_elem_per_row(zxy_grid)
		semantic_map = semantic_map.reshape(L, M)

		semantic_map = semantic_map[self.min_z_coord : self.max_z_coord+1, self.min_x_coord:self.max_x_coord+1]
		color_semantic_map = apply_color_to_map(semantic_map)

		#plt.imshow(color_semantic_map)
		#plt.show()
		if semantic_map.shape[0] > 0:
			save_fig_through_plt(color_semantic_map, f'{self.saved_folder}/step_{step_}_semantic.jpg')


	def save_final_map(self, ENLARGE_SIZE=5):
		# argmax over the category axis
		zyx_grid = np.argmax(self.four_dim_grid, axis=3)
		# swap y dim to the last axis
		zxy_grid = np.swapaxes(zyx_grid, 1, 2)
		L, M, N = zxy_grid.shape
		zxy_grid = zxy_grid.reshape(L*M, N)

		semantic_map = find_first_nonzero_elem_per_row(zxy_grid)
		semantic_map = semantic_map.reshape(L, M)

		semantic_map = semantic_map[self.min_z_coord : self.max_z_coord+1, self.min_x_coord:self.max_x_coord+1]

		map_dict = {}
		map_dict['min_x'] = self.min_x_coord
		map_dict['max_x'] = self.max_x_coord
		map_dict['min_z'] = self.min_z_coord
		map_dict['max_z'] = self.max_z_coord
		map_dict['min_X'] = self.min_X
		map_dict['max_X'] = self.max_X
		map_dict['min_Z'] = self.min_Z
		map_dict['max_Z'] = self.max_Z
		map_dict['W'] = self.W
		map_dict['H'] = self.H
		map_dict['semantic_map'] = semantic_map
		print(f'semantic_map.shape = {semantic_map.shape}')
		np.save(f'{self.saved_folder}/BEV_semantic_map.npy', map_dict)

		semantic_map = cv2.resize(semantic_map, (int(self.W*ENLARGE_SIZE), int(self.H*ENLARGE_SIZE)), interpolation=cv2.INTER_NEAREST)
		color_semantic_map = apply_color_to_map(semantic_map)
		save_fig_through_plt(color_semantic_map, f'{self.saved_folder}/final_semantic_map.jpg')
