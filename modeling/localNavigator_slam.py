import numpy as np, cv2, imageio
import os
import modeling.utils.depth_utils as du
import modeling.utils.rotation_utils as ru
from .utils.fmm_planner import FMMPlanner
import skimage
import matplotlib.pyplot as plt
from core import cfg
from .utils.baseline_utils import pose_to_coords, pxl_coords_to_pose
from math import pi

class localNav_slam(object):
	def __init__(self, pose_range, coords_range, WH, dt=30, mark_locs=False, reset_if_drift=False, count=-1, 
		close_small_openings=False, recover_on_collision=False, fix_thrashing=False, goal_f=1.1, point_cnt=2):

		self.pose_range = pose_range
		self.coords_range = coords_range
		self.WH = WH

		self.dt = dt
		self.count = count
		self.mark_locs = mark_locs
		self.reset_if_drift = reset_if_drift
		self.close_small_openings = close_small_openings
		self.num_erosions = 2
		self.recover_on_collision = recover_on_collision
		self.fix_thrashing = fix_thrashing
		self.goal_f = goal_f
		self.point_cnt = point_cnt

		self.act_trans_dict = {0: 1, 1: 2, 2: 3, 3: 0}
		 
	def reset(self, occ_map, soft=False):
		self.RESET = False

		self.selem = skimage.morphology.disk(int(cfg.SENSOR.AGENT_RADIUS/cfg.SEM_MAP.CELL_SIZE))
		self.selem_small = skimage.morphology.disk(1)
		# 0 agent moves forward. Agent moves in the direction of +x
		# 1 rotates left 
		# 2 rotates right
		# 3 agent stop
		self.loc_on_map = np.zeros(occ_map.shape, dtype=np.float32)

		#self.current_loc = pose_to_coords(agent_map_pose, self.pose_range, self.coords_range, self.WH)

		self.goal_loc = None
		self.last_act = 3
		self.locs = []
		self.acts = []

		if not soft:
			self.num_resets = 0
			self.count = self.count+1
			self.trials = 0
			self.recovery_actions = []
			self.thrashing_actions = []
	
	def plan_to_reach_frontier(self, agent_map_pose, chosen_frontier, occ_map):
		self.current_loc = agent_map_pose
		fron_centroid_coords = (int(chosen_frontier.centroid[1]),
								int(chosen_frontier.centroid[0]))

		subgoal_coords = fron_centroid_coords
		subgoal_pose = pxl_coords_to_pose(subgoal_coords, self.pose_range,
										  self.coords_range, self.WH)

		agent_coords = pose_to_coords(agent_map_pose, self.pose_range, self.coords_range, self.WH)
		self.mark_on_map(agent_coords)
		theta = agent_map_pose[2]
		theta = (theta - .5 * pi)
		state = np.array([agent_coords[0], agent_coords[1], theta])
		
		obstacle = (occ_map == cfg.FE.COLLISION_VAL)
		traversible = skimage.morphology.binary_dilation(obstacle, self.selem) != True
		if self.mark_locs:
			traversible_locs = skimage.morphology.binary_dilation(self.loc_on_map, self.selem) == True
			traversible = np.logical_or(traversible_locs, traversible)
		
		if self.close_small_openings:
			n = self.num_erosions
			reachable = False
			# multi rounds of erosion and dilation
			while n >= 0 and not reachable:
				traversible_open = traversible.copy()
				for i in range(n):
					traversible_open = skimage.morphology.binary_erosion(traversible_open, self.selem_small)
				for i in range(n):
					traversible_open = skimage.morphology.binary_dilation(traversible_open, self.selem_small)
				planner = FMMPlanner(traversible_open, 360//self.dt)
				goal_loc_int = np.array(subgoal_coords).astype(np.int32)
				reachable = planner.set_goal(goal_loc_int)
				reachable = reachable[int(round(state[1])), int(round(state[0]))]
				n = n-1
		else:
			planner = FMMPlanner(traversible, 360//self.dt)
			goal_loc_int = np.array(subgoal_coords).astype(np.int32)
			reachable = planner.set_goal(goal_loc_int)

		self.fmm_dist = planner.fmm_dist * 1
		a, state, act_seq = planner.get_action(state)
		for i in range(len(act_seq)):
			if act_seq[i] == 3:
				act_seq[i] = 0
			elif act_seq[i] == 0:
				act_seq[i] = 3
		if a == 3:
			a = 0
		elif a == 0:
			a = 3

		self.act_seq = act_seq
		self.act_idx = 0

		act = self.act_trans_dict[a]
		return act, act_seq, subgoal_coords, subgoal_pose

	'''
	def next_action(self):
		if len(self.act_seq) == self.act_idx:
			return -1
		act = self.act_seq[self.act_idx]
		act = self.act_trans_dict[act]
		self.act_idx += 1
		return act
	'''

	'''
	def update_loc(self, last_act, pointgoal=None):
		# Currently ignores goal_loc.
		if last_act == 1: # rotate left
			self.current_loc[2] = self.current_loc[2] + self.dt*np.pi/180.
		elif last_act == 2: # rotate right
			self.current_loc[2] = self.current_loc[2] - self.dt*np.pi/180.
		elif last_act == 0:
			self.current_loc[0] = self.current_loc[0] + 25*np.cos(self.current_loc[2])
			self.current_loc[1] = self.current_loc[1] + 25*np.sin(self.current_loc[2])
		self.locs.append(self.current_loc+0)
		self.mark_on_map(self.current_loc)
	'''
	
	def mark_on_map(self, loc):
		# input is the coordinates on the map
		self.loc_on_map[loc[1], loc[0]] = 1
	
	def soft_reset(self, pointgoal):
		# This reset is called if there is drift in the position of the goal
		# location, indicating that there had been collisions.
		if self.out_dir is not None:
			self.save_vis()
		self._reset(pointgoal[0]*100., soft=True)
		self.trials = self.trials+1
		self.num_resets = self.num_resets+1
		xy = self.compute_xy_from_pointnav(pointgoal)
		# self.current_loc has been set inside reset
		self.goal_loc = xy*1
		self.goal_loc[0] = self.goal_loc[0] + self.current_loc[0]
		self.goal_loc[1] = self.goal_loc[1] + self.current_loc[1]
		self.mark_on_map(self.goal_loc)
		self.mark_on_map(self.current_loc)
		if self.num_resets == 6:
			# We don't want to keep resetting. First few resets fix themselves,
			# so do it for later resets.
			num_rots = int(np.round(180 / self.dt))
			self.recovery_actions = [1]*num_rots + [0]*6
		else:
			self.recovery_actions = []
	
	'''
	def check_drift(self, pointgoal):
		xy = self.compute_xy_from_pointnav(pointgoal)
		goal_loc = xy*1
		goal_loc[0] = goal_loc[0] + self.current_loc[0]
		goal_loc[1] = goal_loc[1] + self.current_loc[1]
		# np.set_printoptions(precision=3, suppress=True)
		# print(self.last_act, self.current_loc, goal_loc, self.goal_loc, xy, pointgoal)
		return np.linalg.norm(goal_loc - self.goal_loc) > 5
	'''

	'''
	def check_thrashing(self, n, acts):
		thrashing = False
		if len(acts) > n:
			last_act = acts[-1]
			thrashing = last_act == 1 or last_act == 2
			for i in range(2, n+1):
				if thrashing:
					thrashing = acts[-i] == 3-last_act
					last_act = acts[-i]
				else:
					break
		return thrashing
	'''
	
	def compute_xy_from_pointnav(self, pointgoal):
		xy = np.array([np.cos(pointgoal[1]+self.current_loc[2]), 
									 np.sin(pointgoal[1]+self.current_loc[2])], dtype=np.float32)
		xy = xy*pointgoal[0]*100
		return xy
		
	def act(self):
		if self.RESET:
			self.RESET = False
			return self._act(0, True)
		else:
			return self._act(0, False)
	
	def _act(self, i, done):
		if done:
			self._reset(pointgoal[0]*100.)
			# self.current_loc has been set inside reset
			xy = self.compute_xy_from_pointnav(pointgoal)
			self.goal_loc = xy*1
			self.goal_loc[0] = self.goal_loc[0] + self.current_loc[0]
			self.goal_loc[1] = self.goal_loc[1] + self.current_loc[1]
			self.mark_on_map(self.goal_loc)
			self.mark_on_map(self.current_loc)
		
		self.update_loc(self.last_act)

		'''
		drift = self.check_drift(pointgoal)
		if self.reset_if_drift and drift:
			# import pdb; pdb.set_trace()
			self.soft_reset(pointgoal)
		'''

		act, act_seq = self.plan_path(self.goal_loc)
		
		'''
		if self.recover_on_collision:
			if len(self.recovery_actions) > 0:
				act = self.recovery_actions[0] 
				self.recovery_actions = self.recovery_actions[1:]
		
		thrashing = self.check_thrashing(8, self.acts)
		if thrashing and len(self.thrashing_actions) == 0:
			self.thrashing_actions = act_seq 
			# print(1, self.thrashing_actions)
		
		if self.fix_thrashing:
			if len(self.thrashing_actions) > 0:
				act = self.thrashing_actions[0] 
				self.thrashing_actions = self.thrashing_actions[1:]
				# print(2, self.thrashing_actions)
		'''

		self.acts.append(act)
		self.last_act = act 
		
		return act
	
	
