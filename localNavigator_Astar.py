import numpy as np
import matplotlib.pyplot as plt 
from baseline_utils import pose_to_coords, pxl_coords_to_pose, map_rot_to_planner_rot, planner_rot_to_map_rot
import math
import heapq as hq
from collections import deque
from core import cfg
import networkx as nx

upper_thresh_theta = math.pi / 6
lower_thresh_theta = math.pi / 12

## result is in the range [-pi, pi]
def minus_theta_fn (previous_theta, current_theta):
	result = current_theta - previous_theta
	if result < -math.pi:
		result += 2 * math.pi
	if result > math.pi:
		result -= 2 * math.pi
	return result

def plus_theta_fn (previous_theta, current_theta):
	result = current_theta + previous_theta
	if result < -math.pi:
		result += 2 * math.pi
	if result > math.pi:
		result -= 2 * math.pi
	return result

class Node():
	def __init__(self, loc, parent, cost):
		self.loc = loc
		self.parent = parent # parent node
		#self.action = action # action leads from parent to current node
		self.cost = cost

class TreeList():
	def __init__(self):
		self.list = []

	def searchNode(self, searched_node):
		# check if node loc existed
		flag = False
		for node in self.list:
			if node.loc == searched_node.loc:
				flag = True
		return flag

	def insertNode(self, node):
		self.list.append(node)

	def efficientSearch(self, loc):
		start_idx = len(self.list) - 1
		idx = -1
		for i in range(start_idx, -1, -1):
			if self.list[i].loc == loc:
				idx = i
				break
		assert idx > -1
		node = self.list[idx]
		for i in range(idx-1, -1, -1):
			cur_node = self.list[i]
			if cur_node.loc == node.loc and cur_node.cost <= node.cost:
				node = cur_node
		#print(f'node.loc = {node.loc}, node.cost = {node.cost}')
		return node

	def getNode(self, loc):
		return self.efficientSearch(loc)

	def formPath(self, goal_loc):
		# the last node in the list must be a goal node
		locs = []
		'''
		for i in range(len(self.list)):
			print(f'node {i}, loc = {self.list[i].loc}, cost = {self.list[i].cost}')
		'''
		node = self.efficientSearch(goal_loc)
		
		while True:
			if node.parent is None:
				break
			locs.append(node.loc)
			parent_node = node.parent
			node = parent_node

		return locs[::-1]


class PriorityQueue:
	"""
	  Implements a priority queue data structure. Each inserted item
	  has a priority associated with it and the client is usually interested
	  in quick retrieval of the lowest-priority item in the queue. This
	  data structure allows O(1) access to the lowest-priority item.
	"""
	def  __init__(self):
		self.heap = []
		self.count = 0

	def push(self, item, priority):
		entry = (priority, self.count, item)
		hq.heappush(self.heap, entry)
		self.count += 1

	def pop(self):
		(_, _, item) = hq.heappop(self.heap)
		return item

	def isEmpty(self):
		return len(self.heap) == 0

	def update(self, item, priority):
		# If item already in priority queue with higher priority, update its priority and rebuild the heap.
		# If item already in priority queue with equal or lower priority, do nothing.
		# If item not in priority queue, do the same thing as self.push.
		for index, (p, c, i) in enumerate(self.heap):
			if i == item:
				if p <= priority:
					break
				del self.heap[index]
				self.heap.append((priority, c, item))
				hq.heapify(self.heap)
				break
		else:
			self.push(item, priority)

def breadthFirstSearch(start_coords, graph):
	"""Search the shallowest nodes in the search tree first."""
	visited = []
	Q = deque()
	
	node = start_coords
	Q.append(node)

	while True:
		if len(Q) == 0:
			return visited
		else:
			node = Q.popleft()
			if node in graph:
				# check its successors
				for nei in graph[node]:
					if nei not in visited and nei not in Q:
						Q.append(nei)
			visited.append(node)

def AStarSearch(start_coords, goal_coords, graph):
	tree = TreeList()
	visited = []
	Q = PriorityQueue()

	start_node = Node(start_coords, None, 0.)
	goal_node = Node(goal_coords, None, 0.)
	tree.insertNode(start_node)
	Q.push(start_node.loc, 0)

	while True:
		if Q.isEmpty():
			print(f'failed to find the path ...')
			return [] # fail the search

		node_loc = Q.pop()
		node = tree.getNode(node_loc)
		if (node.loc == goal_coords).all():
			path = tree.formPath(node_loc)
			return path
		else:
			if node_loc in graph:
				for nei in graph[node_loc]:
					dist = math.sqrt((nei[0] - node_loc[0])**2 + (nei[1] - node_loc[1])**2)
					new_node = Node(nei, node, dist + node.cost)
					tree.insertNode(new_node)
					if nei not in visited:
						heur = math.sqrt((nei[0] - goal_coords[0])**2 + (nei[1] - goal_coords[1])**2)
						# update Q
						Q.update(nei, new_node.cost + heur)
				# add node to visited
				visited.append(node_loc)

def build_graph(occupancy_map):
	H, W = occupancy_map.shape
	G = nx.grid_2d_graph(*occupancy_map.shape)

	G.add_edges_from([
	    ((x, y), (x+1, y+1))
	    for x in range(1, H-1)
	    for y in range(1, W-1)
	] + [
	    ((x, y), (x-1, y-1))
	    for x in range(1, H-1)
	    for y in range(1, W-1)
	] + [
	    ((x, y), (x-1, y+1))
	    for x in range(1, H-1)
	    for y in range(1, W-1)
	] + [
	    ((x, y), (x+1, y-1))
	    for x in range(1, H-1)
	    for y in range(1, W-1)
	])

	# remove those nodes where the corresponding value is != 0
	for val, node in zip(occupancy_map.ravel(), sorted(G.nodes())):
	    if val!=cfg.FE.FREE_VAL:
	        G.remove_node(node)

	return G

class localNav_Astar:
	def __init__(self, pose_range, coords_range, WH, scene_name):
		self.pose_range = pose_range
		self.coords_range = coords_range
		self.WH = WH
		self.local_map_margin = cfg.LN.LOCAL_MAP_MARGIN
		self.path_pose_action = []
		self.path_idx = -1 # record the index of the agent in the path
		
		occ_map_path = f'{cfg.SAVE.OCCUPANCY_MAP_PATH}/{scene_name}'
		occupancy_map = np.load(f'{occ_map_path}/BEV_occupancy_map.npy', allow_pickle=True).item()['occupancy']
		occupancy_map = np.where(occupancy_map==1, 3, occupancy_map) # free cell
		occupancy_map = np.where(occupancy_map==0, 1, occupancy_map) # occupied cell
		self.occupancy_map = occupancy_map

	def plan_to_reach_frontier(self, chosen_frontier, agent_pose, occupancy_map, step, saved_folder):
		agent_coords = pose_to_coords(agent_pose, self.pose_range, self.coords_range, self.WH)
		fron_centroid_coords = (int(chosen_frontier.centroid[1]), int(chosen_frontier.centroid[0]))
		#print(f'agent_coords = {agent_coords}')

		#================================ find a reachable subgoal on the map ==============================
		local_occupancy_map = occupancy_map.copy()
		local_occupancy_map[local_occupancy_map == cfg.FE.UNOBSERVED_VAL] = cfg.FE.COLLISION_VAL

		G = build_graph(local_occupancy_map)

		#===================== find the subgoal (closest to peak and reachable from agent)
		subgoal_coords = fron_centroid_coords
		subgoal_pose = pxl_coords_to_pose(subgoal_coords, self.pose_range, self.coords_range, self.WH)

		#============================== Using A* to navigate to the subgoal ==============================
		print(f'agent_coords = {agent_coords[::-1]}, subgoal_coords = {subgoal_coords[::-1]}')
		path = nx.shortest_path(G, source=agent_coords[::-1], target=subgoal_coords[::-1])
		path = [t[::-1] for t in path]

		#========================== visualize the path ==========================
		#'''
		if cfg.LN.FLAG_VISUALIZE_LOCAL_MAP:
			mask_new = mask_free.astype(np.int16)
			for loc in path:
				mask_new[loc[1], loc[0]] = 2
			mask_new[agent_coords[1], agent_coords[0]] = 3 # agent cell
			mask_new[subgoal_coords[1], subgoal_coords[0]] = 4 # subgoal cell
			mask_new[fron_centroid_coords[1], fron_centroid_coords[0]] = 5 # peak cell

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
			pose = pxl_coords_to_pose((loc[0], loc[1]), self.pose_range, self.coords_range, self.WH)
			points.append(pose)

		## compute theta for each point except the last one
		## theta is in the range [-pi, pi]
		thetas = []
		for i in range(len(points) - 1):
			p1 = points[i]
			p2 = points[i + 1]
			current_theta = math.atan2(p2[1]-p1[1], p2[0]-p1[0])
			thetas.append(current_theta)

		#print(f'len(thetas) = {len(thetas)}, len(points) = {len(points)}')
		assert len(thetas) == len(points) - 1

		# pose: (x, y, theta)
		previous_theta = 0
		for i in range(len(points) - 1):
			p1 = points[i]
			p2 = points[i+1]

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
			while abs(minus_theta_fn(previous_theta, current_theta)) > upper_thresh_theta:
				if bool_turn:
					previous_theta = minus_theta_fn(upper_thresh_theta, previous_theta)
					actions.append("TURN_RIGHT")
				else:
					previous_theta = plus_theta_fn(upper_thresh_theta, previous_theta)
					actions.append("TURN_LEFT")
				pose = (p1[0], p1[1], previous_theta)
				poses.append(pose)
			## add one more turning points when change of theta > 15 degree
			if abs(minus_theta_fn(previous_theta, current_theta)) > lower_thresh_theta:
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
				action = actions[i-1]
			self.path_pose_action.append((new_pose, action))

		#print(f'path_idx = {self.path_idx}, path_pose_action = {self.path_pose_action}')
		return True, subgoal_coords, subgoal_pose

	def next_action(self, env, height):
		if self.path_idx >= len(self.path_pose_action):
			return "", None

		pose, action = self.path_pose_action[self.path_idx]

		self.path_idx += 1

		agent_pose = (pose[0], height, pose[1])
		if not env.habitat_env.sim.is_navigable(agent_pose):
			action = 'collision'
			print(f'collision pose = {agent_pose}')
		return action, pose

	def _decide_local_map_size(self, agent_coords, subgoal_coords, H, W):
		x1, z1 = agent_coords
		x2, z2 = subgoal_coords

		xmin = min(x1, x2)
		xmax = max(x1, x2)
		zmin = min(z1, z2)
		zmax = max(z1, z2)

		xmin = max(0, xmin - self.local_map_margin)
		xmax = min(W, xmax + self.local_map_margin)
		zmin = max(0, zmin - self.local_map_margin)
		zmax = min(H, zmax + self.local_map_margin)

		agent_local_coords = (agent_coords[0] - xmin, agent_coords[1] - zmin)
		subgoal_local_coords = (subgoal_coords[0] - xmin, subgoal_coords[1] - zmin)
		#agent_local_coords = agent_coords
		#subgoal_local_coords = subgoal_coords
		#xmin = 0
		#zmin = 0

		return (xmin, zmin, xmax, zmax), agent_local_coords, subgoal_local_coords

	def filter_unreachable_frontiers(self, frontiers, agent_pose, occupancy_map):
		agent_coords = pose_to_coords(agent_pose, self.pose_range, self.coords_range, self.WH)
		#print(f'agent_coords = {agent_coords}')

		#================================ find a reachable subgoal on the map ==============================
		local_occupancy_map = occupancy_map.copy()
		local_occupancy_map[local_occupancy_map == cfg.FE.UNOBSERVED_VAL] = cfg.FE.COLLISION_VAL

		G = build_graph(local_occupancy_map)

		reachable_locs = list(nx.node_connected_component(G, (agent_coords[1], agent_coords[0])))
		reachable_locs = [t[::-1] for t in reachable_locs]
		
		filtered_frontiers = set()
		for fron in frontiers:
			fron_centroid_coords = (int(fron.centroid[1]), int(fron.centroid[0]))
			if fron_centroid_coords in reachable_locs:
				filtered_frontiers.add(fron)
		return filtered_frontiers


''' test
LN = localNav_Astar((-10.6, -17.5, 18.4, 10.6), (91, 159, 198, 258))
agent_map_pose = (6.6, 6.9, 2.36)
subgoal = np.array([84, 44], dtype=np.int16)
occupancy_map = np.load('local_occupancy_map.npy')
LN.plan(agent_map_pose, subgoal, occupancy_map)
'''
