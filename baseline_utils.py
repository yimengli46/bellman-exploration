import collections
import copy
import json
import os
import networkx as nx
import numpy as np
import numpy.linalg as LA
import scipy.io as sio
import cv2
import math
from math import cos, sin, acos, atan2, pi, floor, tan
from io import StringIO
import matplotlib.pyplot as plt
from constants import coco_categories_mapping, panopticSeg_mapping, d3_41_colors_rgb, COCO_74_COLORS
import matplotlib as mpl
from core import cfg

def minus_theta_fn(previous_theta, current_theta):
  result = current_theta - previous_theta
  if result < -math.pi:
    result += 2*math.pi
  if result > math.pi:
    result -= 2*math.pi
  return result

def project_pixels_to_camera_coords (sseg_img, current_depth, current_pose, gap=2, FOV=90, cx=320, cy=240, resolution_x=640, resolution_y=480, ignored_classes=[]):
  ## camera intrinsic matrix
  FOV = 79
  radian = FOV * pi / 180.
  focal_length = cx/tan(radian/2)
  K = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]])
  inv_K = LA.inv(K)
  ## first compute the rotation and translation from current frame to goal frame
  ## then compute the transformation matrix from goal frame to current frame
  ## thransformation matrix is the camera2's extrinsic matrix
  tx, tz, theta = current_pose
  R = np.array([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])
  T = np.array([tx, 0, tz])
  transformation_matrix = np.empty((3, 4))
  transformation_matrix[:3, :3] = R
  transformation_matrix[:3, 3] = T
  
  # build the point matrix
  x = range(0, resolution_x, gap)
  y = range(0, resolution_y, gap)
  xv, yv = np.meshgrid(np.array(x), np.array(y))
  Z = current_depth[yv.flatten(), xv.flatten()].reshape(yv.shape[0], yv.shape[1])
  points_4d = np.ones((yv.shape[0], yv.shape[1], 4), np.float32)
  points_4d[:, :, 0] = xv
  points_4d[:, :, 1] = yv
  points_4d[:, :, 2] = Z
  points_4d = np.transpose(points_4d, (2, 0, 1)).reshape((4, -1)) # 4 x N

  # apply intrinsic matrix
  points_4d[[0, 1, 3], :] = inv_K.dot(points_4d[[0, 1, 3], :])
  points_4d[0, :] = points_4d[0, :] * points_4d[2, :]
  points_4d[1, :] = points_4d[1, :] * points_4d[2, :]

  ## transform kp1_4d from camera1(current) frame to camera2(goal) frame through transformation matrix
  print('points_4d.shape = {}'.format(points_4d.shape))
  points_3d = points_4d[:3, :]
  print('points_3d.shape = {}'.format(points_3d.shape))

  ## pick x-row and z-row
  sseg_points = sseg_img[yv.flatten(), xv.flatten()].flatten()

  # ignore some classes points
  #print('sseg_points.shape = {}'.format(sseg_points.shape))
  for c in ignored_classes:
    good = (sseg_points != c)
    sseg_points = sseg_points[good]
    points_3d = points_3d[:, good]
  #print('after: sseg_points.shape = {}'.format(sseg_points.shape))
  #print('after: points_3d.shape = {}'.format(points_3d.shape))

  return points_3d, sseg_points.astype(int)


def project_pixels_to_world_coords (sseg_img, current_depth, current_pose, gap=2, FOV=79, cx=320, cy=240, theta_x=0.0, resolution_x=640, resolution_y=480, ignored_classes=[]):
  ## camera intrinsic matrix
  radian = FOV * pi / 180.
  focal_length = cx/tan(radian/2)
  K = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]])
  inv_K = LA.inv(K)
  ## first compute the rotation and translation from current frame to goal frame
  ## then compute the transformation matrix from goal frame to current frame
  ## thransformation matrix is the camera2's extrinsic matrix
  tx, tz, theta = current_pose
  #theta = -(theta + 0.5 * pi)
  #theta = -theta
  R_y = np.array([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])
  # used when I tilt the camera up/down
  R_x = np.array([[1, 0, 0], [0, cos(theta_x), -sin(theta_x)], [0, sin(theta_x), cos(theta_x)]])
  R = R_y.dot(R_x)
  T = np.array([tx, 0, tz])
  transformation_matrix = np.empty((3, 4))
  transformation_matrix[:3, :3] = R
  transformation_matrix[:3, 3] = T
  
  # build the point matrix
  x = range(0, resolution_x, gap)
  y = range(0, resolution_y, gap)
  xv, yv = np.meshgrid(np.array(x), np.array(y))
  Z = current_depth[yv.flatten(), xv.flatten()].reshape(yv.shape[0], yv.shape[1])
  points_4d = np.ones((yv.shape[0], yv.shape[1], 4), np.float32)
  points_4d[:, :, 0] = xv
  points_4d[:, :, 1] = yv
  points_4d[:, :, 2] = Z
  points_4d = np.transpose(points_4d, (2, 0, 1)).reshape((4, -1)) # 4 x N

  # apply intrinsic matrix
  points_4d[[0, 1, 3], :] = inv_K.dot(points_4d[[0, 1, 3], :])
  points_4d[0, :] = points_4d[0, :] * points_4d[2, :]
  points_4d[1, :] = points_4d[1, :] * points_4d[2, :]

  ## transform kp1_4d from camera1(current) frame to camera2(goal) frame through transformation matrix
  points_3d = transformation_matrix.dot(points_4d)

  # ignore some artifacts points with depth == 0
  depth_points = current_depth[yv.flatten(), xv.flatten()].flatten()
  good = np.logical_and(depth_points > .5, depth_points < 5)
  #print(f'points_3d.shape = {points_3d.shape}')
  points_3d = points_3d[:, good]
  #print(f'points_3d.shape = {points_3d.shape}')

  ## pick x-row and z-row
  sseg_points = sseg_img[yv.flatten(), xv.flatten()].flatten()
  sseg_points = sseg_points[good]

  # ignore some classes points
  #print('sseg_points.shape = {}'.format(sseg_points.shape))
  for c in ignored_classes:
    good = (sseg_points != c)
    sseg_points = sseg_points[good]
    points_3d = points_3d[:, good]
  #print('after: sseg_points.shape = {}'.format(sseg_points.shape))
  #print('after: points_3d.shape = {}'.format(points_3d.shape))  

  return points_3d, sseg_points.astype(int)

def convertInsSegToSSeg (InsSeg, scene_graph_npz, cat2id_dict):
  ins_id_list = list(scene_graph_npz['object'].keys())
  SSeg = np.zeros(InsSeg.shape, dtype=np.int32)
  for ins_id in ins_id_list:
    cat_id = cat2id_dict[scene_graph_npz['object'][ins_id]['class_']]
    SSeg = np.where(InsSeg==ins_id, cat_id, SSeg)

  return SSeg

def convertMaskRCNNToSSeg (detectron2_npy, H=480, W=640, det_thresh=0.5):
  #print(detectron2_npy)
  SSeg = np.zeros((H, W), dtype=np.int32) # 15 semantic categories
  idxs = list(range(len(detectron2_npy['classes'])))
  #print(f'idxs = {idxs}')
  for j in idxs[::-1]:
    class_idx = detectron2_npy['classes'][j]
    score = detectron2_npy['scores'][j]
    #print(f'j = {j}, class = {class_idx}')
    if class_idx in list(coco_categories_mapping.keys()) and score > det_thresh:
      idx = coco_categories_mapping[class_idx] + 1 # first class has index 0
      obj_mask = detectron2_npy['masks'][j]
      SSeg = np.where(obj_mask, idx, SSeg)
  return SSeg

def convertPanopSegToSSeg (PanopSeg, id2cat_dict):
  SSeg = np.zeros(PanopSeg.shape, dtype=np.int32)
  for cat_id in list(panopticSeg_mapping.keys()):
    mapped_cat_id = panopticSeg_mapping[cat_id]
    SSeg = np.where(PanopSeg==cat_id, mapped_cat_id, SSeg)

  return SSeg

# if # of classes is <= 41, flag_small_categories is True
def apply_color_to_map (semantic_map, flag_small_categories=False):
  assert len(semantic_map.shape) == 2
  if flag_small_categories:
    COLOR = d3_41_colors_rgb
    num_classes = 41
  else:
    COLOR = COCO_74_COLORS
    num_classes = 74

  H, W = semantic_map.shape
  color_semantic_map = np.zeros((H, W, 3), dtype='uint8')
  for i in range(num_classes):
    color_semantic_map[semantic_map==i] = COLOR[i]
  return color_semantic_map

def apply_color_to_pointCloud (sseg_points, num_classes=41):
  assert len(sseg_points.shape) == 1
  N = sseg_points.shape[0]
  color_sseg_points = np.zeros((N, 3), dtype='uint8')
  for i in range(num_classes):
    color_sseg_points[sseg_points==i] = d3_41_colors_rgb[i]
  return color_sseg_points

def create_folder (folder_name, clean_up=False):
  flag_exist = os.path.isdir(folder_name)
  if not flag_exist:
    print('{} folder does not exist, so create one.'.format(folder_name))
    os.makedirs(folder_name)
    #os.makedirs(os.path.join(test_case_folder, 'observations'))
  else:
    print('{} folder already exists, so do nothing.'.format(folder_name))
    if clean_up:
      os.system('rm {}/*.png'.format(folder_name))
      os.system('rm {}/*.npy'.format(folder_name))
      os.system('rm {}/*.jpg'.format(folder_name))

def read_map_npy(map_npy):
  min_x = map_npy['min_x']
  max_x = map_npy['max_x']
  min_z = map_npy['min_z']
  max_z = map_npy['max_z']
  min_X = map_npy['min_X']
  max_X = map_npy['max_X']
  min_Z = map_npy['min_Z']
  max_Z = map_npy['max_Z']
  semantic_map = map_npy['semantic_map']
  return semantic_map, (min_X, min_Z, max_X, max_Z), (min_x, min_z, max_x, max_z)

def semanticMap_to_binary(sem_map):
  sem_map.astype('uint8')
  sem_map[sem_map != 2] = 0
  sem_map[sem_map == 2] = 255
  return sem_map

def get_class_mapper(dataset='gibson'):
  class_dict = {}
  if dataset == 'mp3d':
    categories = ['void', 'wall', 'floor','chair','door','table','picture','cabinet','cushion','window','sofa','bed', \
      'curtain','chest_of_drawers','plant','sink','stairs','ceiling','toilet','stool','towel','mirror','tv_monitor', \
      'shower','column','bathtub','counter','fireplace','lighting','beam','railing','shelving','blinds','gym_equipment', \
      'seating','board_panel','furniture','appliances','clothes','objects','misc']
  elif dataset == 'gibson':
    categories = list(np.load(f'{cfg.PF.SEMANTIC_PRIOR_PATH}/all_objs_list.npy', allow_pickle=True))
  class_dict = {v: k+1 for k, v in enumerate(categories)}
  return class_dict

def get_room_class_mapper(dataset='gibson'):
  room_list = np.load(f'{cfg.PF.SEMANTIC_PRIOR_PATH}/room_type_list.npy', allow_pickle=True)
  class_dict = {v: k+1 for k, v in enumerate(room_list)}
  return class_dict

def pxl_coords_to_pose(coords, pose_range, coords_range, cell_size=0.1, flag_cropped=True):
  x, y = coords
  min_X, min_Z, max_X, max_Z = pose_range
  min_x, min_z, max_x, max_z = coords_range

  if flag_cropped:
    X = (x + 0.5 + min_x) * cell_size + min_X
    Z = (y + 0.5 + min_z) * cell_size + min_Z
  else:
    X = (x + 0.5) * cell_size + min_X
    Z = (y + 0.5) * cell_size + min_Z
  return (X, Z)

def pxl_coords_to_pose_numpy(coords, pose_range, coords_range, cell_size=0.1, flag_cropped=True):
  min_X, min_Z, max_X, max_Z = pose_range
  min_x, min_z, max_x, max_z = coords_range

  pose = np.zeros(coords.shape)
  if flag_cropped:
    pose[:, 0] = (coords[:, 0] + min_x) * cell_size + min_X
    pose[:, 1] = (coords[:, 1] + min_z) * cell_size + min_Z
  else:
    pose[:, 0] = (coords[:, 0]) * cell_size + min_X
    pose[:, 1] = (coords[:, 1]) * cell_size + min_Z
  return pose


def pose_to_coords(cur_pose, pose_range, coords_range, cell_size=0.1, flag_cropped=True):
  tx, tz = cur_pose[:2]
    
  if flag_cropped:
    x_coord = int(floor((tx - pose_range[0]) / cell_size) - coords_range[0])
    z_coord = int(floor((tz - pose_range[1]) / cell_size) - coords_range[1])
  else:
    x_coord = int(floor((tx - pose_range[0]) / cell_size))
    z_coord = int(floor((tz - pose_range[1]) / cell_size))

  return (x_coord, z_coord)

def pose_to_coords_numpy(cur_pose, pose_range, coords_range, cell_size=0.1, flag_cropped=True):    
  coords = np.zeros(cur_pose.shape)
  if flag_cropped:
    coords[:, 0] = (np.floor((cur_pose[:, 0] - pose_range[0]) / cell_size) - coords_range[0]).astype(int)
    coords[:, 1] = (np.floor((cur_pose[:, 1] - pose_range[1]) / cell_size) - coords_range[1]).astype(int)
  else:
    coords[:, 0] = np.floor((cur_pose[:, 0] - pose_range[0]) / cell_size)
    coords[:, 1] = np.floor((cur_pose[:, 1] - pose_range[1]) / cell_size)

  coords = coords.astype(int)
  return coords

# for particle visualization only
def pose_to_coords_frame(cur_pose, pose_range, coords_range, cell_size=0.1, flag_cropped=True):
  tx, tz = cur_pose[:2]
    
  if flag_cropped:
    x_coord = (tx - pose_range[0]) / cell_size - coords_range[0]
    z_coord = (tz - pose_range[1]) / cell_size - coords_range[1]
    if x_coord < 0:
      x_coord = 0.
    if z_coord < 0:
      z_coord = 0.
    coords_0_max = coords_range[2] - coords_range[0]
    coords_1_max = coords_range[3] - coords_range[1]
    if x_coord > coords_0_max:
      x_coord = coords_0_max
    if z_coord > coords_1_max:
      z_coord = coords_1_max
  else:
    x_coord = (tx - pose_range[0]) / cell_size
    z_coord = (tz - pose_range[1]) / cell_size

  return (x_coord, z_coord)

# for particle visualization only
def pose_to_coords_frame_numpy(cur_pose, pose_range, coords_range, cell_size=0.1, flag_cropped=True):    
  coords = np.zeros(cur_pose.shape, dtype=np.float32)
  if flag_cropped:
    coords[:, 0] = (cur_pose[:, 0] - pose_range[0]) / cell_size - coords_range[0]
    coords[:, 1] = (cur_pose[:, 1] - pose_range[1]) / cell_size - coords_range[1]
    mask = (coords[:, 0] < 0.)
    coords[mask, 0] = 0.
    mask = (coords[:, 1] < 0.)
    coords[mask, 1] = 0.
    coords_0_max = coords_range[2] - coords_range[0]
    coords_1_max = coords_range[3] - coords_range[1]
    mask = (coords[:, 0] > coords_0_max)
    coords[mask, 0] = coords_0_max
    mask = (coords[:, 1] > coords_1_max)
    coords[mask, 1] = coords_1_max
  else:
    coords[:, 0] = (cur_pose[:, 0] - pose_range[0]) / cell_size
    coords[:, 1] = (cur_pose[:, 1] - pose_range[1]) / cell_size

  return coords

def save_fig_through_plt(img, name):
  fig, ax = plt.subplots(nrows=1, ncols=1)
  ax.imshow(img)
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  fig.tight_layout()
  #plt.show()
  fig.savefig(name)
  plt.close()


def gen_arrow_head_marker(rot):
    """generate a marker to plot with matplotlib scatter, plot, ...

    https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers

    rot=0: positive x direction
    Parameters
    ----------
    rot : float
        rotation in radian
        0 is positive x direction

    Returns
    -------
    arrow_head_marker : Path
        use this path for marker argument of plt.scatter
    scale : float
        multiply a argument of plt.scatter with this factor got get markers
        with the same size independent of their rotation.
        Paths are autoscaled to a box of size -1 <= x, y <= 1 by plt.scatter
    """

    # rotate the rot to the marker's coordinate system
    rotate_rot = rot - .5*pi
    #print(f'rot in drawing is {math.degrees(rot)}, rotate_rot is {math.degrees(rotate_rot)}')
    rot = math.degrees(rotate_rot)
    #print(f'visualized angle = {rot}')

    arr = np.array([[.1, .3], [.1, -.3], [1, 0]])  # arrow shape
    angle = rot / 180 * np.pi
    rot_mat = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
        ])
    arr = np.matmul(arr, rot_mat)  # rotates the arrow

    # scale
    x0 = np.amin(arr[:, 0])
    x1 = np.amax(arr[:, 0])
    y0 = np.amin(arr[:, 1])
    y1 = np.amax(arr[:, 1])
    scale = np.amax(np.abs([x0, x1, y0, y1]))

    arrow_head_marker = mpl.path.Path(arr)
    return arrow_head_marker, scale

def map_rot_to_planner_rot(rot):
  rotate_rot = -rot + .5 * pi
  return rotate_rot

def planner_rot_to_map_rot(rot):
  rotate_rot = -(rot - .5 * pi)
  return rotate_rot