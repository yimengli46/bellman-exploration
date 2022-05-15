import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import math
import torch

color_mapping_3 = {
	0:(255,255,255), # white
	1:(0,0,255), # blue
	2:(0,255,0), # green
}


def colorize_grid(grid, color_mapping=3): # to pass into tensorboardX video
	# Input: grid -- B x T x C x grid_dim x grid_dim, where C=1,T=1 when gt and C=41,T>=1 for other
	# Output: grid_img -- B x T x 3 x grid_dim x grid_dim
	grid = grid.detach().cpu().numpy()
	grid_img = np.zeros((grid.shape[0], grid.shape[1], grid.shape[3], grid.shape[4], 3),  dtype=np.uint8)
	if grid.shape[2] > 1:
		# For cells where prob distribution is all zeroes (or uniform), argmax returns arbitrary number (can be true for the accumulated maps)
		grid_prob_max = np.amax(grid, axis=2)
		inds = np.asarray(grid_prob_max<=0.33).nonzero() # if no label has prob higher than k then assume unobserved
		grid[inds[0], inds[1], 0, inds[2], inds[3]] = 1 # assign label 0 (void) to be the dominant label
		grid = np.argmax(grid, axis=2) # B x T x grid_dim x grid_dim
	else:
		grid = grid.squeeze(2)

	color_mapping = color_mapping_3
	for label in color_mapping.keys():
		grid_img[ grid==label ] = color_mapping[label]
	
	return torch.tensor(grid_img.transpose(0, 1, 4, 2, 3), dtype=torch.uint8)