import numpy as np
import numpy.linalg as LA
import cv2
import math
import matplotlib.patches as patches
import networkx as nx
import random
import habitat

def change_brightness(img, flag, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    #lim = 255 - value
    #v[v > lim] = 255
    #v[v <= lim] += value

    v[np.logical_and(flag == False, v > value)] -= value
    v[np.logical_and(flag == False, v <= value)] = 0

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

class SimpleRLEnv(habitat.RLEnv):
	def get_reward_range(self):
		return [-1, 1]

	def get_reward(self, observations):
		return 0

	def get_done(self, observations):
		return self.habitat_env.episode_over

	def get_info(self, observations):
		return self.habitat_env.get_metrics()


def get_scene_name(episode):
	idx_right_most_slash = episode.scene_id.rfind('/')
	return episode.scene_id[idx_right_most_slash+1:-4]

def verify_img(img):
	sum_img = np.sum((img[:,:,0] > 0))
	h, w = img.shape[:2]
	return sum_img > h*w*0.75