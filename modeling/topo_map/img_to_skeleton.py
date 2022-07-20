import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import feature
from skimage.morphology import skeletonize
from scipy import ndimage
import scipy.stats as st

"""Returns a 2D Gaussian kernel array."""
def point_gaussian(kernel_size=21, sigma=3):
	kernel_size = int(kernel_size)
	# construct input size
	interval = (2 * sigma + 1.) / kernel_size
	x = np.linspace(-sigma-interval/2., sigma+interval/2., kernel_size+1)
	# 1-d cdf on input x
	kernel_1d = np.diff(st.norm.cdf(x))
	# outer product, 1-d gaussian filter to 2-d
	kernel_raw = np.sqrt(np.outer(kernel_1d, kernel_1d))
	kernel = kernel_raw / np.max(kernel_raw)#.sum()
	return kernel

def map_gaussian(im, kernel_size, sigma):
	h, w = im.shape

	point_gs = point_gaussian(kernel_size, sigma)
	map_gs = np.zeros((h, w), dtype=np.float)

	kernel_radius = (kernel_size - 1) / 2

	occ_loc = np.where(im[:, :] == 0)
	#print('occ_loc = {}'.format(occ_loc))
	#assert 1==2
	for i in range(len(occ_loc[1])):
		loc_i = [occ_loc[0][i], occ_loc[1][i]]
		# map_gs bound
		lower_bd_y, lower_bd_x = int(max(0, loc_i[0]-kernel_radius)), int(max(0, loc_i[1]-kernel_radius))
		upper_bd_y, upper_bd_x = int(min(h-1, loc_i[0]+kernel_radius)), int(min(w-1, loc_i[1]+kernel_radius))
		# kernel bound
		k_lower_bd_y, k_lower_bd_x = int(-min(0, loc_i[0]-kernel_radius)), int(-min(0, loc_i[1]-kernel_radius))
		k_upper_bd_y, k_upper_bd_x = int(kernel_size+(h-1 - max(h-1, loc_i[0]+kernel_radius))), int(kernel_size + (w-1 - max(w-1, loc_i[1]+kernel_radius)))
		# maximum on pixel
		map_gs[lower_bd_y:upper_bd_y+1, lower_bd_x:upper_bd_x+1] = \
			np.maximum(map_gs[lower_bd_y:upper_bd_y+1, lower_bd_x:upper_bd_x+1], 
			point_gs[k_lower_bd_y:k_upper_bd_y, k_lower_bd_x:k_upper_bd_x])

	return map_gs 

def img2skeleton(im):
	#1. gaussian map
	im, map_gs = preprocess(im)
	#2. edge detection
	dst = edgeprocess(map_gs)
	#3. binarize
	res = binarize(dst, 10)
	#4. suppress
	skeleton = skeletonize(res) 
	#5. remove T shape center
	skeleton = remove_Tcenter(skeleton)

	return im, map_gs, skeleton

def preprocess(im):
	#im = cv2.imread(im_path, 0)
	h, w = im.shape
	if h < 1000 and w < 1000:
		print('resize img, original image is too small ...')
		im = cv2.resize(im, (int(w*10), int(h*10)), interpolation=cv2.INTER_NEAREST)
	print('im.shape = {}'.format(im.shape))

	# using gaussian kernel
	map_gs = map_gaussian(im, 455, 5)
	return im, map_gs

def edgeprocess(gs):
	#laplacian
	gray_lap = cv2.Laplacian(gs*255, cv2.CV_64F, ksize=5)	
	dst = cv2.convertScaleAbs(gray_lap)
	return dst

def binarize(dst, thresh):
	res = np.zeros(dst.shape, dtype=np.float)
	res[np.where(dst > thresh)] = 1
	return res

Tidu = (np.array([0, 1, 1, 1]),np.array([1, 0, 1, 2]))
Tidd = (np.array([1, 1, 1, 2]),np.array([0, 1, 2, 1]))
Tidl = (np.array([0, 1, 1, 2]),np.array([1, 0, 1, 1]))
Tidr = (np.array([0, 1, 1, 2]),np.array([1, 1, 2, 1]))

def remove_Tcenter(skele):
	'''
		it is possible to have T shape in skeleton, we should remove the center of it
	'''
	w,h = skele.shape
	for i in range(1, w-1):
		for j in range(1, h-1):
			if skele[i, j]:
				patch = skele[i-1:i+2, j-1:j+2].copy()
				if np.sum(patch[Tidu]) == 4 or \
					np.sum(patch[Tidd]) == 4 or \
					np.sum(patch[Tidl]) == 4 or \
					np.sum(patch[Tidr]) == 4: 
					skele[i,j] = False
	#i = 0
	for j in range(1, h-1):
		if skele[0, j]:
			if np.sum(skele[0, j-1:j+2]) + skele[1, j] == 4:
				skele[0, j] = False
	#i = w-1
	for j in range(1, h-1):
		if skele[w-1, j]:
			if np.sum(skele[w-1, j-1:j+2]) + skele[w-2, j] == 4:
				skele[w-1, j] = False
	#j = 0
	for i in range(1, w-1):
		if skele[i, 0]:
			if np.sum(skele[i-1:i+2, 0]) + skele[i, 1] == 4:
				skele[i, 0] = False
	#j = h-1
	for i in range(1, w-1):
		if skele[i, h-1]:
			if np.sum(skele[i-1:i+2, h-1]) + skele[i, h-2] == 4:
				skele[i, h-1] = False
 
	#i=0,j=0
	if skele[0, 0]:
		if skele[0, 0] + skele[0, 1] + skele[1, 0] == 3:
			skele[0, 0] = False
	#i=0,j=h-1
	if skele[0, h-1]:
		if skele[0, h-2] + skele[0, h-1] + skele[1, h-1] == 3:
			skele[0, h-1] = False
	#i=w-1,j=0
	if skele[w-1, 0]:
		if skele[w-2, 0] + skele[w-1, 0] + skele[w-1, 1] == 3:
			skele[w-1, 0] = False
	#i=w-1,j=h-1
	if skele[w-1, h-1]:
		if skele[w-1, h-2] + skele[w-1, h-1] + skele[w-2, h-1] == 3:
			skele[w-1, h-1] = False
	return skele