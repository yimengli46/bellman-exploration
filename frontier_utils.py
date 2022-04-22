import numpy as np 
import matplotlib.pyplot as plt
from core import cfg
import scipy.ndimage
import numpy as np 
import matplotlib.pyplot as plt
from core import cfg
import scipy.ndimage
from baseline_utils import pose_to_coords
from math import sqrt

class Frontier(object):
	def __init__(self, points):
		"""Initialized with a 2xN numpy array of points (the grid cell
		coordinates of all points on frontier boundary)."""
		inds = np.lexsort((points[0, :], points[1, :]))
		sorted_points = points[:, inds]
		self.props_set = False
		self.is_from_last_chosen = False
		self.is_obstructed = False
		self.prob_feasible = 1.0
		self.delta_success_cost = 0.0
		self.exploration_cost = 0.0
		self.negative_weighting = 0.0
		self.positive_weighting = 0.0

		self.counter = 0
		self.last_observed_pose = None

		# Any duplicate points should be eliminated (would interfere with
		# equality checking).
		dupes = []
		for ii in range(1, sorted_points.shape[1]):
			if (sorted_points[:, ii - 1] == sorted_points[:, ii]).all():
				dupes += [ii]
		self.points = np.delete(sorted_points, dupes, axis=1)

		# Compute and cache the hash
		self.hash = hash(self.points.tobytes())

		self.R = 1
		self.D = 1.

	def set_props(self,
				  prob_feasible,
				  is_obstructed=False,
				  delta_success_cost=0,
				  exploration_cost=0,
				  positive_weighting=0,
				  negative_weighting=0,
				  counter=0,
				  last_observed_pose=None,
				  did_set=True):
		self.props_set = did_set
		self.just_set = did_set
		self.prob_feasible = prob_feasible
		self.is_obstructed = is_obstructed
		self.delta_success_cost = delta_success_cost
		self.exploration_cost = exploration_cost
		self.positive_weighting = positive_weighting
		self.negative_weighting = negative_weighting
		self.counter = counter
		self.last_observed_pose = last_observed_pose

	@property
	def centroid(self):
		#return self.get_centroid()
		return self.get_frontier_point()

	#'''
	def get_centroid(self):
		"""Returns the point that is the centroid of the frontier"""
		centroid = np.mean(self.points, axis=1)
		return centroid
	#'''

	'''
	def get_centroid(self):
		#print(f'points.shape = {self.points.shape}')
		points = self.points.transpose()
		distMatrix = np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2, axis=-1)
		centroid_idx = np.argmin(distMatrix.sum(axis=0))
		centroid = self.points[:, centroid_idx]
		return centroid
	'''

	def get_frontier_point(self):
		"""Returns the point that is on the frontier that is closest to the
		actual centroid"""
		center_point = np.mean(self.points, axis=1)
		norm = np.linalg.norm(self.points - center_point[:, None], axis=0)
		ind = np.argmin(norm)
		return self.points[:, ind]

	def get_distance_to_point(self, point):
		norm = np.linalg.norm(self.points - point[:, None], axis=0)
		return norm.min()

	def __hash__(self):
		return self.hash

	def __eq__(self, other):
		return hash(self) == hash(other)

def mask_grid_with_frontiers(occupancy_grid, frontiers, do_not_mask=None):
	"""Mask grid cells in the provided occupancy_grid with the frontier points
	contained with the set of 'frontiers'. If 'do_not_mask' is provided, and
	set to either a single frontier or a set of frontiers, those frontiers are
	not masked."""

	if do_not_mask is not None:
		# Ensure that 'do_not_mask' is a set
		if isinstance(do_not_mask, Frontier):
			do_not_mask = set([do_not_mask])
		elif not isinstance(do_not_mask, set):
			raise TypeError("do_not_mask must be either a set or a Frontier")
		masking_frontiers = frontiers - do_not_mask
	else:
		masking_frontiers = frontiers

	masked_grid = occupancy_grid.copy()
	for frontier in masking_frontiers:
		masked_grid[frontier.points[0, :],
					frontier.points[1, :]] = 2

	return masked_grid

def get_frontiers(occupancy_grid, gt_occupancy_grid, observed_area_flag):
	filtered_grid = scipy.ndimage.maximum_filter(occupancy_grid == cfg.FE.UNOBSERVED_VAL, size=3)
	frontier_point_mask = np.logical_and(filtered_grid, occupancy_grid == cfg.FE.FREE_VAL)

	if cfg.FE.GROUP_INFLATION_RADIUS < 1:
		inflated_frontier_mask = frontier_point_mask
	else:
		inflated_frontier_mask = gridmap.utils.inflate_grid(frontier_point_mask,
			inflation_radius=cfg.FE.GROUP_INFLATION_RADIUS, obstacle_threshold=0.5,
			collision_val=1.0) > 0.5

	# Group the frontier points into connected components
	labels, nb = scipy.ndimage.label(inflated_frontier_mask)

	# Extract the frontiers
	frontiers = set()
	for ii in range(nb):
		raw_frontier_indices = np.where(np.logical_and(labels == (ii + 1), frontier_point_mask))
		frontiers.add(
			Frontier(
				np.concatenate((raw_frontier_indices[0][None, :],
								raw_frontier_indices[1][None, :]),
							   axis=0)))

	# Compute potential
	if cfg.NAVI.PERCEPTION == 'Potential':
		free_but_unobserved_flag = np.logical_and(gt_occupancy_grid == cfg.FE.FREE_VAL, observed_area_flag == False)
		free_but_unobserved_flag = scipy.ndimage.maximum_filter(free_but_unobserved_flag, size=3)

		labels, nb = scipy.ndimage.label(free_but_unobserved_flag)

		for ii in range(nb):
			component = (labels == (ii+1))
			for f in frontiers:
				if component[int(f.centroid[0]), int(f.centroid[1])]:
					f.R = np.sum(component)
					f.D = round(sqrt(f.R), 2)

					if cfg.NAVI.FLAG_VISUALIZE_FRONTIER_POTENTIAL:
						fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
						ax[0].imshow(occupancy_grid)
						ax[0].scatter(f.points[1], f.points[0], c='white', zorder=2)
						ax[0].scatter(f.centroid[1], f.centroid[0], c='red', zorder=2)
						ax[0].get_xaxis().set_visible(False)
						ax[0].get_yaxis().set_visible(False)
						ax[0].set_title('explored occupancy map')

						ax[1].imshow(component)
						ax[1].get_xaxis().set_visible(False)
						ax[1].get_yaxis().set_visible(False)
						ax[1].set_title('area potential')

						ax[2].imshow(gt_occupancy_grid)
						ax[2].get_xaxis().set_visible(False)
						ax[2].get_yaxis().set_visible(False)
						ax[2].set_title('gt occupancy map')

						fig.tight_layout()
						plt.title(f'component {ii}')
						plt.show()

	return frontiers

def remove_isolated_points(occupancy_grid, threshold=2):
	H, W = occupancy_grid.shape
	new_grid = occupancy_grid.copy()
	for i in range(1, H-1):
		for j in range(1, W-1):
			if occupancy_grid[i][j] == cfg.FE.UNOBSERVED_VAL:
				new_grid[i][j] = nearest_value_og(occupancy_grid, i, j, threshold=threshold)
	return new_grid

def nearest_value_og(occupancy_grid, i, j, threshold=4):
	d = {cfg.FE.COLLISION_VAL:0, cfg.FE.FREE_VAL:0, cfg.FE.UNOBSERVED_VAL:0}
	d[occupancy_grid[i-1][j]] += 1
	d[occupancy_grid[i+1][j]] += 1
	d[occupancy_grid[i][j-1]] += 1
	d[occupancy_grid[i][j+1]] += 1
	  
	for occupancy_value, count in d.items():
		if count >= threshold:
			return occupancy_value
	return occupancy_grid[i][j]

def get_frontier_with_maximum_area(frontiers, gt_occupancy_grid):
	if cfg.NAVI.PERCEPTION == 'Anticipation':
		count_free_space_at_frontiers(frontiers, gt_occupancy_grid)
		max_area = 0
		max_fron = None
		for fron in frontiers:
			if fron.area_neigh > max_area:
				max_area = fron.area_neigh
				max_fron = fron
	elif cfg.NAVI.PERCEPTION == 'Potential':
		max_area = 0
		max_fron = None
		for fron in frontiers:
			if fron.R > max_area:
				max_area = fron.R
				max_fron = fron

	return max_fron


def count_free_space_at_frontiers(frontiers, gt_occupancy_grid, area=10):
	H, W = gt_occupancy_grid.shape
	for fron in frontiers:
		centroid = (int(fron.centroid[1]), int(fron.centroid[0]))
		x1 = max(0, centroid[0] - area)
		x2 = min(W, centroid[0] + area)
		y1 = max(0, centroid[1] - area)
		y2 = min(H, centroid[1] + area)
		fron_neigh = gt_occupancy_grid[y1:y2, x1:x2]
		#print(f'centroid[0] = {centroid[0]}, y1 = {y1}, y2= {y2}, x1 = {x1}, x2 = {x2}')
		#plt.imshow(fron_neigh)
		#plt.show()
		fron.area_neigh = np.sum(fron_neigh == cfg.FE.FREE_VAL)
		#print(f'fron.area_neigh = {fron.area_neigh}')

def get_frontier_with_DP(frontiers, agent_pose, observed_occupancy_map, steps, LN):
	max_Q = 0
	max_frontier = None
	G = LN.get_G_from_map(observed_occupancy_map)
	agent_coord = LN.get_agent_coords(agent_pose)
	steps = steps / 3.

	for fron in frontiers:
		#print('-------------------------------------------------------------')
		visited_frontiers = set()
		Q = compute_Q(agent_coord, fron, frontiers, visited_frontiers, steps, G, LN)
		if Q >= max_Q:
			max_Q = Q
			max_frontier = fron
	return max_frontier

def compute_Q(agent_coord, target_frontier, frontiers, visited_frontiers, steps, G, LN):
	#print(f'agent_coord = {agent_coord}, target_frontier = {target_frontier.centroid}, steps = {steps}')
	Q = 0
	L = LN.compute_L(G, agent_coord, target_frontier)

	# cond 1: agent has enough steps to reach target_frontier
	if steps > L:
		steps -= L

		# cond 2: agent does not have enough steps to traverse target_frontier:
		if steps <= target_frontier.D:
			Q += 1. * steps / target_frontier.D * target_frontier.R
		else:
			steps -= target_frontier.D
			Q += target_frontier.R
			# cond 3: agent does have enough steps to reach target_frontier
			if steps >= target_frontier.D:
				steps -= target_frontier.D
				visited_frontiers.add(target_frontier)
				rest_frontiers = frontiers - visited_frontiers

				max_next_Q = 0
				for fron in rest_frontiers:
					fron_centroid_coords = (int(target_frontier.centroid[1]), int(target_frontier.centroid[0]))
					next_Q = compute_Q(fron_centroid_coords, fron, frontiers, visited_frontiers.copy(), steps, G, LN)
					if next_Q > max_next_Q:
						max_next_Q = next_Q
				Q += max_next_Q
	#print(f'Q = {Q}')
	return Q