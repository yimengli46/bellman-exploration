import os
import numpy as np
import matplotlib.pyplot as plt
from core import cfg
import bz2
import _pickle as cPickle

folder = 'output/training_data_input_view_1000samples/train/17DRP5sb8fy_0'

with bz2.BZ2File(f'{folder}/087.pbz2', 'rb') as fp:
	npy_file = cPickle.load(fp)

	panorama_rgb = npy_file['rgb']
	panorama_depth = npy_file['depth']
	panorama_sseg = npy_file['sseg']
	
	fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))
	ax[0].imshow(panorama_rgb)
	ax[0].get_xaxis().set_visible(False)
	ax[0].get_yaxis().set_visible(False)
	ax[0].set_title("rgb")
	ax[1].imshow(panorama_sseg)
	ax[1].get_xaxis().set_visible(False)
	ax[1].get_yaxis().set_visible(False)
	ax[1].set_title("sseg")
	ax[2].imshow(panorama_depth)
	ax[2].get_xaxis().set_visible(False)
	ax[2].get_yaxis().set_visible(False)
	ax[2].set_title("depth")
	fig.tight_layout()
	plt.show()