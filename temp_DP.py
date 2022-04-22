from math import sqrt

class fron(object):
	def __init__(self, R, D, center):
		self.R = R
		self.D = D 
		self.center = center

def compute_Q(agent_coord, target_frontier, frontiers, visited_frontiers, steps):
	#print(f'agent_coord = {agent_coord}, target_frontier = {target_frontier.center}, steps = {steps}')
	Q = 0
	L = abs(agent_coord[0] - target_frontier.center[0]) + abs(agent_coord[1] - target_frontier.center[1])

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

					next_Q = compute_Q(target_frontier.center, fron, frontiers, visited_frontiers.copy(), steps)
					if next_Q > max_next_Q:
						max_next_Q = next_Q
				Q += max_next_Q
	#print(f'Q = {Q}')
	return Q


frontiers = set()
f1 = fron(50, 50, (10, 10))
f2 = fron(20, 20, (15, 15))
f3 = fron(10, 10, (20, 20))

frontiers.add(f1)
frontiers.add(f2)
frontiers.add(f3)

agent_start = (0, 0)
steps = 200



for steps in list(range(10, 300, 10)):
	max_Q = 0
	max_frontier = None
	for fron in frontiers:
		print('-------------------------------------------------------------')
		visited_frontiers = set()
		Q = compute_Q(agent_start, fron, frontiers, visited_frontiers, steps)
		if Q >= max_Q:
			max_Q = Q
			max_frontier = fron

	print(f'steps = {steps}, max_frontier = {max_frontier.center}')

