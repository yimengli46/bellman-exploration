#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np

from habitat.tasks.utils import cartesian_to_polar, quaternion_rotate_vector
import gzip
import json
import matplotlib.pyplot as plt
import quaternion as qt
import math

def get_scene_name(episode):
    idx_right_most_slash = episode.scene_id.rfind('/')
    return episode.scene_id[idx_right_most_slash+1:-4]

split = 'test'
saved_folder = '../output/scene_height_distribution'

scene_start_y_dict = {}
scene_height_dict = {}

filename = f'../data/habitat_data/datasets/pointnav/mp3d/v1/{split}/{split}.json.gz'
with gzip.open(filename , 'rb') as f:
    data = json.loads(f.read())
episodes = data['episodes']

#============================= summarize the start point y of each scene =========================
for episode in episodes:
    scene_id = episode['scene_id']

    pos_slash = scene_id.rfind('/')
    pos_dot = scene_id.rfind('.')
    episode_scene = scene_id[pos_slash+1:pos_dot]

    start_pose_y = episode['start_position'][1]

    if episode_scene in scene_start_y_dict:
        scene_start_y_dict[episode_scene].append(start_pose_y)
    else:
        scene_start_y_dict[episode_scene] = [start_pose_y]

for scene_name in list(scene_start_y_dict.keys()):
    
    height_lst = scene_start_y_dict[scene_name]

    values, counts = np.unique(height_lst, return_counts=True)

    scene_height_dict[scene_name] = {}
    scene_height_dict[scene_name]['values'] = values
    scene_height_dict[scene_name]['counts'] = counts
    
#================================ summarize the y values of each scene =========================
# only take y with more than 5 counts
scene_floor_dict = {}
thresh_counts = 5

for scene_name in list(scene_height_dict.keys()):
    values = scene_height_dict[scene_name]['values']
    counts = scene_height_dict[scene_name]['counts']

    scene_floor_dict[scene_name] = {}

    idx_max = np.argmax(counts)
    scene_floor_dict[scene_name][0] = {}
    scene_floor_dict[scene_name][0]['y'] = values[idx_max]

    '''
    # find the y for each floor
    count_floor = 0
    for idx, val in enumerate(values):
        count = counts[idx]
        if count >= thresh_counts:
            scene_floor_dict[scene_name][count_floor] = {}
            scene_floor_dict[scene_name][count_floor]['y'] = val
            count_floor += 1
    '''

gap_thresh = 0.01
for episode in episodes:
    scene_id = episode['scene_id']

    pos_slash = scene_id.rfind('/')
    pos_dot = scene_id.rfind('.')
    scene_name = scene_id[pos_slash+1:pos_dot]

    start_pose_y = episode['start_position'][1]

    for idx_floor in list(scene_floor_dict[scene_name].keys()):
        floor_y = scene_floor_dict[scene_name][idx_floor]['y']
        if start_pose_y - floor_y < gap_thresh:
            x = episode['start_position'][0]
            z = episode['start_position'][2]

            a, b, c, d = episode['start_rotation']
            agent_rot = qt.quaternion(a,b,c,d)
            heading_vector = quaternion_rotate_vector(agent_rot.inverse(), np.array([0, 0, -1]))
            phi = round(cartesian_to_polar(-heading_vector[2], heading_vector[0])[1], 4)

            pose = (x, z, phi)

            if 'start_pose' in scene_floor_dict[scene_name][idx_floor]:
                scene_floor_dict[scene_name][idx_floor]['start_pose'].append(pose)
            else:
                scene_floor_dict[scene_name][idx_floor]['start_pose'] = [pose]

            break

np.save(f'{saved_folder}/{split}_scene_floor_dict.npy', scene_floor_dict)