import numpy as np 
import matplotlib.pyplot as plt 

robot_center = (700, 800)

local_map = np.zeros((480, 480))
local_map[0:240, 0:240] = 5
local_map[240:480, 0:240] = 6
local_map[0:240, 240:480] = 7
local_map[240:480, 240:480] = 8

global_map = np.ones((1000, 1000))

def inter_local_map_global_map(local_map, global_map, robot_center):
	H_local, W_local = local_map.shape
	H_global, W_global = global_map.shape

	left_corner_local = np.array((0, 0))
	right_corner_local = np.array((W_local-1, H_local-1))
	left_corner_global = np.array((0, 0))
	right_corner_global = np.array((W_global-1, H_global-1))

	# move local map whose center is now at robot center
	robot_center = np.array(robot_center)
	local_map_center = np.array((W_local//2, H_local//2))
	trans = robot_center - local_map_center
	left_corner_local += trans
	right_corner_local += trans

	print(f'local: {left_corner_local}, {right_corner_local}')

	# find intersection
	x0_global = max(left_corner_local[0], left_corner_global[0])
	x1_global = min(right_corner_local[0], right_corner_global[0])
	y0_global = max(left_corner_local[1], left_corner_global[1])
	y1_global = min(right_corner_local[1], right_corner_global[1])

	# move bbox back to local map coords
	x0_local, y0_local = np.array((x0_global, y0_global)) - trans
	x1_local, y1_local = np.array((x1_global, y1_global)) - trans

	return np.array((x0_local, y0_local, x1_local, y1_local)), np.array((x0_global, y0_global, x1_global, y1_global))


bbox_local, bbox_global = inter_local_map_global_map(local_map, global_map, robot_center)

global_map[bbox_global[1]:bbox_global[3]+1, bbox_global[0]:bbox_global[2]+1] = local_map[bbox_local[1]:bbox_local[3]+1, bbox_local[0]:bbox_local[2]+1]

plt.imshow(global_map)
plt.show()