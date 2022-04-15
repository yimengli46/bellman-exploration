import numpy as np 
import matplotlib.pyplot as plt
from core import cfg
import scipy.ndimage
import numpy as np 
import matplotlib.pyplot as plt
from core import cfg
import scipy.ndimage
from baseline_utils import pose_to_coords

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
		return self.get_centroid()

	def get_centroid(self):
		"""Returns the point that is the centroid of the frontier"""
		centroid = np.mean(self.points, axis=1)
		return centroid

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

def get_frontiers(occupancy_grid):
	filtered_grid = scipy.ndimage.maximum_filter(occupancy_grid == cfg.FE.FREE_VAL, size=3)
	frontier_point_mask = np.logical_and(filtered_grid, occupancy_grid == cfg.FE.UNOBSERVED_VAL)

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

def get_frontier_with_maximum_area(frontiers, visited_frontiers, gt_occupancy_grid):
	count_free_space_at_frontiers(frontiers, gt_occupancy_grid)
	max_area = 0
	max_fron = None
	for fron in frontiers:
		if fron not in visited_frontiers:
			if fron.area_neigh > max_area:
				max_area = fron.area_neigh
				max_fron = fron

	return max_fron


def count_free_space_at_frontiers(frontiers, gt_occupancy_grid, area=10):
	H, W = gt_occupancy_grid.shape
	for fron in frontiers:
		centroid = (int(fron.centroid[1]), int(fron.centroid[0]))
		x1 = max(0, centroid[1] - area)
		x2 = min(W, centroid[1] + area)
		y1 = max(0, centroid[0] - area)
		y2 = min(H, centroid[0] + area)
		fron_neigh = gt_occupancy_grid[y1:y2, x1:x2]
		fron.area_neigh = np.sum(fron_neigh == cfg.FE.FREE_VAL)


