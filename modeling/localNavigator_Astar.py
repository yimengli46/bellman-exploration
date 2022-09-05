import numpy as np
import matplotlib.pyplot as plt
from .utils.baseline_utils import pose_to_coords, pxl_coords_to_pose, map_rot_to_planner_rot, planner_rot_to_map_rot, minus_theta_fn, plus_theta_fn
import math
import heapq as hq
from collections import deque
from core import cfg
import networkx as nx
from timeit import default_timer as timer
import scipy.ndimage

upper_thresh_theta = math.pi / 6
lower_thresh_theta = math.pi / 12


def build_graph(occupancy_map, flag_eight_neighs=True):
	"""
	Convert the grid-like occupancy_map into a graph G through networkx.
	Each node in the graph corresponds to a free cell in the occupancy map.
	Each node has 8 neighbors.
	"""
	#t1 = timer()
	H, W = occupancy_map.shape
	G = nx.grid_2d_graph(H, W)
	#t2 = timer()
	#print(f'**** grid_2d_graph time = {t2 - t1}')

	if flag_eight_neighs:
		for edge in G.edges:
			G.edges[edge]['weight'] = 1
		G.add_edges_from([((x, y), (x + 1, y + 1)) for x in range(0, H - 1)
						  for y in range(0, W - 1)] + [((x + 1, y), (x, y + 1))
													   for x in range(0, H - 1)
													   for y in range(0, W - 1)], weight=1.4)

	# remove those nodes where map is occupied
	mask_occupied_node = (occupancy_map.ravel() == cfg.FE.COLLISION_VAL)
	nodes_npy = np.array(sorted(G.nodes))
	nodes_occupied = nodes_npy[mask_occupied_node]
	lst_nodes_occupied = list(map(tuple, nodes_occupied))
	#t3 = timer()
	#print(f'**** get occupied nodes list time = {t3 - t2}')
	
	G.remove_nodes_from(lst_nodes_occupied)

	#t4 = timer()
	#print(f'**** remove nodes from time = {t4 - t3}')

	return G


class localNav_Astar:

	def __init__(self, pose_range, coords_range, WH, scene_name=None):
		self.pose_range = pose_range
		self.coords_range = coords_range
		self.WH = WH
		self.local_map_margin = cfg.LN.LOCAL_MAP_MARGIN
		self.path_pose_action = []
		self.path_idx = -1  # record the index of the agent in the path

	def plan_to_reach_frontier(self, chosen_frontier, agent_pose,
							   occupancy_map, step, saved_folder):
		"""Plan a path from agent_pose to the chosen_frontier on the occupancy_map.
		step and saved_folder is for saving visualizations during the planning.

		We first convert the occupancy_map into a graph G.
		Then use A* to plan a path from the agent_pose to the chosen_frontier.

		In the end, we convert the path (a list of cell locations) into a sequence of actions
		including turning and moving forward.
		"""
		agent_coords = pose_to_coords(agent_pose, self.pose_range,
									  self.coords_range, self.WH)
		fron_centroid_coords = (int(chosen_frontier.centroid[1]),
								int(chosen_frontier.centroid[0]))
		#print(f'agent_coords = {agent_coords}')

		#================================ find a reachable subgoal on the map ==============================
		local_occupancy_map = occupancy_map.copy()
		local_occupancy_map[local_occupancy_map ==
							cfg.FE.UNOBSERVED_VAL] = cfg.FE.COLLISION_VAL

		G = build_graph(local_occupancy_map)

		#===================== find the subgoal (closest to peak and reachable from agent)
		subgoal_coords = fron_centroid_coords
		subgoal_pose = pxl_coords_to_pose(subgoal_coords, self.pose_range,
										  self.coords_range, self.WH)

		#============================== Using A* to navigate to the subgoal ==============================
		print(
			f'agent_coords = {agent_coords[::-1]}, subgoal_coords = {subgoal_coords[::-1]}'
		)
		path = nx.shortest_path(G,
								source=agent_coords[::-1],
								target=subgoal_coords[::-1])
		path = [t[::-1] for t in path]

		#========================== visualize the path ==========================
		#'''
		if cfg.LN.FLAG_VISUALIZE_LOCAL_MAP:
			mask_new = mask_free.astype(np.int16)
			for loc in path:
				mask_new[loc[1], loc[0]] = 2
			mask_new[agent_coords[1], agent_coords[0]] = 3  # agent cell
			mask_new[subgoal_coords[1], subgoal_coords[0]] = 4  # subgoal cell
			mask_new[fron_centroid_coords[1],
					 fron_centroid_coords[0]] = 5  # peak cell

			fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
			# visualize gt semantic map
			ax[0].imshow(local_occupancy_map)
			ax[0].get_xaxis().set_visible(False)
			ax[0].get_yaxis().set_visible(False)
			ax[0].set_title('local_occupancy_map')
			# visualize built semantic map
			ax[1].imshow(mask_new, vmin=0, vmax=5)
			ax[1].get_xaxis().set_visible(False)
			ax[1].get_yaxis().set_visible(False)
			ax[1].set_title('planned path')
			plt.show()
			#fig.savefig(f'{saved_folder}/step_{step}_localPlanner.jpg')
			#plt.close()

		#'''

		#============================== convert path to poses ===================
		print(f'len(path) = {len(path)}')
		if len(path) <= 1:
			print(f'agent already reached subgoal')
			return False, subgoal_coords, subgoal_pose

		poses = []
		actions = []
		points = []

		for loc in path:
			pose = pxl_coords_to_pose((loc[0], loc[1]), self.pose_range,
									  self.coords_range, self.WH)
			points.append(pose)

		## compute theta for each point except the last one
		## theta is in the range [-pi, pi]
		thetas = []
		for i in range(len(points) - 1):
			p1 = points[i]
			p2 = points[i + 1]
			current_theta = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
			thetas.append(current_theta)

		#print(f'len(thetas) = {len(thetas)}, len(points) = {len(points)}')
		assert len(thetas) == len(points) - 1

		# pose: (x, y, theta)
		previous_theta = 0
		for i in range(len(points) - 1):
			p1 = points[i]
			p2 = points[i + 1]

			current_theta = thetas[i]
			## so that previous_theta is same as current_theta for the first point
			if i == 0:
				previous_theta = map_rot_to_planner_rot(agent_pose[2])
			#print(f'previous_theta = {math.degrees(previous_theta)}, current_theta = {math.degrees(current_theta)}')
			## first point is not the result of an action
			## append an action before introduce a new pose
			if i != 0:
				## forward: 0, left: 3, right 2
				actions.append("MOVE_FORWARD")
			## after turning, previous theta is changed into current_theta
			pose = (p1[0], p1[1], previous_theta)
			poses.append(pose)
			## first add turning points
			## decide turn left or turn right, Flase = left, True = Right
			bool_turn = False
			minus_cur_pre_theta = minus_theta_fn(previous_theta, current_theta)
			if minus_cur_pre_theta < 0:
				bool_turn = True
			## need to turn more than once, since each turn is 30 degree
			while abs(minus_theta_fn(previous_theta,
									 current_theta)) > upper_thresh_theta:
				if bool_turn:
					previous_theta = minus_theta_fn(upper_thresh_theta,
													previous_theta)
					actions.append("TURN_RIGHT")
				else:
					previous_theta = plus_theta_fn(upper_thresh_theta,
												   previous_theta)
					actions.append("TURN_LEFT")
				pose = (p1[0], p1[1], previous_theta)
				poses.append(pose)
			## add one more turning points when change of theta > 15 degree
			if abs(minus_theta_fn(previous_theta,
								  current_theta)) > lower_thresh_theta:
				if bool_turn:
					actions.append("TURN_RIGHT")
				else:
					actions.append("TURN_LEFT")
				pose = (p1[0], p1[1], current_theta)
				poses.append(pose)
			## no need to change theta any more
			previous_theta = current_theta
			## then add forward points

			## we don't need to add p2 to poses unless p2 is the last point in points
			if i + 1 == len(points) - 1:
				actions.append("MOVE_FORWARD")
				pose = (p2[0], p2[1], current_theta)
				poses.append(pose)

		assert len(poses) == (len(actions) + 1)

		self.path_idx = 1
		self.path_pose_action = []
		for i in range(0, len(poses)):
			pose = poses[i]
			# convert planner pose to environment pose
			rot = -planner_rot_to_map_rot(pose[2])
			new_pose = (pose[0], -pose[1], rot)
			if i == 0:
				action = ""
			else:
				action = actions[i - 1]
			self.path_pose_action.append((new_pose, action))

		#print(f'path_idx = {self.path_idx}, path_pose_action = {self.path_pose_action}')
		return True, subgoal_coords, subgoal_pose

	def next_action(self, env, height):
		"""compute the next action and reached pose.
		if a collision occurs after taking the action, action is 'collision.'
		
		input env and height is used to check if the next action will incur a collision.
		"""
		if self.path_idx >= len(self.path_pose_action):
			return "", None

		pose, action = self.path_pose_action[self.path_idx]

		self.path_idx += 1

		agent_pose = (pose[0], height, pose[1])
		if not env.habitat_env.sim.is_navigable(agent_pose):
			action = 'collision'
			print(f'collision pose = {agent_pose}')
		return action, pose

	def filter_unreachable_frontiers(self, frontiers, agent_pose,
									 occupancy_map):
		""" remove the unreachable frontiers from current agent_pose given the occupancy_map.

		The idea is to compute the connected components of agent_pose as 'reachable_locs' on the occupancy_map through BFS.
		If the center of frontier is included in the connected component, this frontier is kept.

		"""
		agent_coords = pose_to_coords(agent_pose, self.pose_range,
									  self.coords_range, self.WH)
		#print(f'agent_coords = {agent_coords}')

		#t1 = timer()
		binary_occupancy_map = occupancy_map.copy()
		binary_occupancy_map[binary_occupancy_map == cfg.FE.UNOBSERVED_VAL] = cfg.FE.COLLISION_VAL
		binary_occupancy_map[binary_occupancy_map == cfg.FE.COLLISION_VAL] = 0
		#t2 = timer()
		#print(f'====> get local map time = {t2 - t1}')

		#t3 = timer()
		#print(f'====> build graph time = {t3 - t2}')

		labels, nb = scipy.ndimage.label(binary_occupancy_map, structure=np.ones((3,3)))
		agent_label = labels[agent_coords[1], agent_coords[0]]
		#t4 = timer()
		#print(f'====> get connected components time = {t4 - t3}')

		filtered_frontiers = set()
		for fron in frontiers:
			fron_centroid_coords = (int(fron.centroid[1]),
									int(fron.centroid[0]))
			fron_label = labels[fron_centroid_coords[1], fron_centroid_coords[0]]
			if fron_label == agent_label:
				filtered_frontiers.add(fron)
		#t5 = timer()
		#print(f'====> filter frontiers time = {t5 - t4}')

		binary_occupancy_map[binary_occupancy_map != 0] = 1
		binary_occupancy_map[binary_occupancy_map == 0] = 1000

		return filtered_frontiers, binary_occupancy_map

	def filter_unreachable_frontiers_temp(self, frontiers, agent_coords,
										  occupancy_map):
		""" remove the unreachable frontiers from current agent_coords given the occupancy_map.

		The idea is to compute the connected components of agent_pose as 'reachable_locs' on the occupancy_map through BFS.
		If the center of frontier is included in the connected component, this frontier is kept.

		"""
		binary_occupancy_map = occupancy_map.copy()
		binary_occupancy_map[binary_occupancy_map == cfg.FE.UNOBSERVED_VAL] = cfg.FE.COLLISION_VAL
		binary_occupancy_map[binary_occupancy_map == cfg.FE.COLLISION_VAL] = 0

		labels, nb = scipy.ndimage.label(binary_occupancy_map, structure=np.ones((3,3)))
		agent_label = labels[agent_coords[1], agent_coords[0]]

		filtered_frontiers = set()
		for fron in frontiers:
			fron_centroid_coords = (int(fron.centroid[1]),
									int(fron.centroid[0]))
			fron_label = labels[fron_centroid_coords[1], fron_centroid_coords[0]]
			if fron_label == agent_label:
				filtered_frontiers.add(fron)
		return filtered_frontiers

	def get_G_from_map(self, occupancy_map):
		""" convert the occupancy_map to a graph G.

		All the unknown cells on the occupancy map are treated as occupied.
		"""
		#================================ find a reachable subgoal on the map ==============================
		local_occupancy_map = occupancy_map.copy()
		local_occupancy_map[local_occupancy_map ==
							cfg.FE.UNOBSERVED_VAL] = cfg.FE.COLLISION_VAL

		G = build_graph(local_occupancy_map)
		return G

	def get_agent_coords(self, agent_pose):
		"""get the agent coordinates on the occupancy map given the agent_pose in the environment.
		"""
		agent_coords = pose_to_coords(agent_pose, self.pose_range,
									  self.coords_range, self.WH)
		return agent_coords

	def compute_L(self, G, agent_coords, frontier):
		""" compute the L in the Bellman Equation as the path length from agent_coords to the frontier on graph G.

		"""
		fron_centroid_coords = (int(frontier.centroid[1]),
								int(frontier.centroid[0]))

		#===================== find the subgoal (closest to peak and reachable from agent)
		subgoal_coords = fron_centroid_coords

		#============================== Using A* to navigate to the subgoal ==============================
		#print(f'agent_coords = {agent_coords[::-1]}, subgoal_coords = {subgoal_coords[::-1]}')
		path = nx.shortest_path(G,
								source=agent_coords[::-1],
								target=subgoal_coords[::-1])
		path = [t[::-1] for t in path]

		return len(path)

	def convert_path_to_pose(self, path):
		path = [t[::-1] for t in path]

		poses = []
		actions = []
		points = []

		for loc in path:
			pose = pxl_coords_to_pose((loc[0], loc[1]), self.pose_range,
									  self.coords_range, self.WH)
			points.append(pose)

		## compute theta for each point except the last one
		## theta is in the range [-pi, pi]
		thetas = []
		for i in range(len(points) - 1):
			p1 = points[i]
			p2 = points[i + 1]
			current_theta = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
			thetas.append(current_theta)

		#print(f'len(thetas) = {len(thetas)}, len(points) = {len(points)}')
		assert len(thetas) == len(points) - 1

		# pose: (x, y, theta)
		previous_theta = 0
		for i in range(len(points) - 1):
			p1 = points[i]
			p2 = points[i + 1]

			current_theta = thetas[i]
			## so that previous_theta is same as current_theta for the first point
			if i == 0:
				previous_theta = map_rot_to_planner_rot(0)
			#print(f'previous_theta = {math.degrees(previous_theta)}, current_theta = {math.degrees(current_theta)}')
			## first point is not the result of an action
			## append an action before introduce a new pose
			if i != 0:
				## forward: 0, left: 3, right 2
				actions.append("MOVE_FORWARD")
			## after turning, previous theta is changed into current_theta
			pose = (p1[0], p1[1], previous_theta)
			poses.append(pose)
			## first add turning points
			## decide turn left or turn right, Flase = left, True = Right
			bool_turn = False
			minus_cur_pre_theta = minus_theta_fn(previous_theta, current_theta)
			if minus_cur_pre_theta < 0:
				bool_turn = True
			## need to turn more than once, since each turn is 30 degree
			while abs(minus_theta_fn(previous_theta,
									 current_theta)) > upper_thresh_theta:
				if bool_turn:
					previous_theta = minus_theta_fn(upper_thresh_theta,
													previous_theta)
					actions.append("TURN_RIGHT")
				else:
					previous_theta = plus_theta_fn(upper_thresh_theta,
												   previous_theta)
					actions.append("TURN_LEFT")
				pose = (p1[0], p1[1], previous_theta)
				poses.append(pose)
			## add one more turning points when change of theta > 15 degree
			if abs(minus_theta_fn(previous_theta,
								  current_theta)) > lower_thresh_theta:
				if bool_turn:
					actions.append("TURN_RIGHT")
				else:
					actions.append("TURN_LEFT")
				pose = (p1[0], p1[1], current_theta)
				poses.append(pose)
			## no need to change theta any more
			previous_theta = current_theta
			## then add forward points

			## we don't need to add p2 to poses unless p2 is the last point in points
			if i + 1 == len(points) - 1:
				actions.append("MOVE_FORWARD")
				pose = (p2[0], p2[1], current_theta)
				poses.append(pose)

		assert len(poses) == (len(actions) + 1)

		path_idx = 1
		pose_lst = []
		for i in range(0, len(poses)):
			pose = poses[i]
			# convert planner pose to environment pose
			rot = -planner_rot_to_map_rot(pose[2])
			new_pose = (pose[0], -pose[1], rot)
			if i == 0:
				action = ""
			else:
				action = actions[i - 1]
			pose_lst.append(new_pose)

		#print(f'path_idx = {self.path_idx}, path_pose_action = {self.path_pose_action}')
		return pose_lst

	def get_start_pose_connected_component(self, agent_pose, occupancy_map):
		agent_coords = pose_to_coords(agent_pose, self.pose_range,
									  self.coords_range, self.WH)

		binary_occupancy_map = occupancy_map.copy()

		labels, nb = scipy.ndimage.label(binary_occupancy_map, structure=np.ones((3,3)))
		agent_label = labels[agent_coords[1], agent_coords[0]]
		
		result = (labels == agent_label)

		return result