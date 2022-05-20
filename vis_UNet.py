import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from UNet import UNet
from dataloader_MP3D import MP3DDataset
import torch.nn.functional as F
from modeling.utils.baseline_utils import apply_color_to_map
from core import cfg
import torch.utils.data as data
from itertools import islice

dataset_val = MP3DDataset(split='train', scene_names=cfg.MAIN.TRAIN_SCENE_LIST, worker_size=0, seed=cfg.GENERAL.RANDOM_SEED, num_elems=10000)
dataloader_val = data.DataLoader(dataset_val, batch_size=1, num_workers=0)

device = torch.device('cuda')

model = UNet(n_channel_in=2, n_class_out=1).to(device)
checkpoint = torch.load(f'run/MP3D/unet/experiment_1/checkpoint.pth.tar')
model.load_state_dict(checkpoint['state_dict'])



with torch.no_grad():
	idx = 0
	for batch in islice(dataloader_val, 1000):
		print(f'idx = {idx}')
		images, targets = batch[0], batch[1]
		#print('images = {}'.format(images))
		#print('targets = {}'.format(targets))
		images, targets = images.cuda(), targets.cuda()

		output = model(images)

		images = images.cpu().numpy()[0]
		targets = targets.cpu().numpy()[0, 0]
		output = output.cpu().numpy()[0, 0]

		occ_map_Mp = images[0]
		sem_map_Mp = images[1]
		color_sem_map_Mp = apply_color_to_map(sem_map_Mp)
		
		fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
		ax[0][0].imshow(occ_map_Mp, cmap='gray')
		ax[0][0].get_xaxis().set_visible(False)
		ax[0][0].get_yaxis().set_visible(False)
		ax[0][0].set_title('input: occupancy_map_Mp')
		ax[0][1].imshow(color_sem_map_Mp)
		ax[0][1].get_xaxis().set_visible(False)
		ax[0][1].get_yaxis().set_visible(False)
		ax[0][1].set_title('input: semantic_map_Mp')
		ax[1][0].imshow(targets, vmin=0.0, vmax=1.0)
		ax[1][0].get_xaxis().set_visible(False)
		ax[1][0].get_yaxis().set_visible(False)
		ax[1][0].set_title('label: U_a')
		ax[1][1].imshow(output, vmin=0.0, vmax=1.0)
		ax[1][1].get_xaxis().set_visible(False)
		ax[1][1].get_yaxis().set_visible(False)
		ax[1][1].set_title('predict: U_a')

		fig.tight_layout()
		#plt.show()
		fig.savefig(f'{cfg.PRED.SAVED_FOLDER}/img_{idx}.jpg')
		plt.close()

		mask_zero = (targets == 0)
		output[mask_zero] = 0

		fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
		ax[0][0].imshow(occ_map_Mp, cmap='gray')
		ax[0][0].get_xaxis().set_visible(False)
		ax[0][0].get_yaxis().set_visible(False)
		ax[0][0].set_title('input: occupancy_map_Mp')
		ax[0][1].imshow(color_sem_map_Mp)
		ax[0][1].get_xaxis().set_visible(False)
		ax[0][1].get_yaxis().set_visible(False)
		ax[0][1].set_title('input: semantic_map_Mp')
		ax[1][0].imshow(targets, vmin=0.0, vmax=1.0)
		ax[1][0].get_xaxis().set_visible(False)
		ax[1][0].get_yaxis().set_visible(False)
		ax[1][0].set_title('label: U_a')
		ax[1][1].imshow(output, vmin=0.0, vmax=1.0)
		ax[1][1].get_xaxis().set_visible(False)
		ax[1][1].get_yaxis().set_visible(False)
		ax[1][1].set_title('predict: U_a')

		fig.tight_layout()
		#plt.show()
		fig.savefig(f'{cfg.PRED.SAVED_FOLDER}/img_{idx}_zero_out.jpg')
		plt.close()

		idx += 1



