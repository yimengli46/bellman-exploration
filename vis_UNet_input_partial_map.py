import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from modeling.utils.UNet import UNet
from dataloader_MP3D import MP3DDataset, my_collate
import torch.nn.functional as F
from modeling.utils.baseline_utils import apply_color_to_map
from core import cfg
import torch.utils.data as data
from itertools import islice
import cv2

Total_Samples = 1000
BATCH_SIZE = 4

dataset_val = MP3DDataset(split='val', scene_names=cfg.MAIN.VAL_SCENE_LIST, worker_size=1, seed=cfg.GENERAL.RANDOM_SEED, num_elems=10000)
dataloader_val = data.DataLoader(dataset_val, batch_size=BATCH_SIZE, num_workers=1, collate_fn=my_collate)

device = torch.device('cuda')

model = UNet(n_channel_in=cfg.PRED.PARTIAL_MAP.INPUT_CHANNEL, n_class_out=cfg.PRED.PARTIAL_MAP.OUTPUT_CHANNEL).to(device)
checkpoint = torch.load(f'run/MP3D/unet/experiment_5/checkpoint.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

with torch.no_grad():
	count = 0
	for batch in islice(dataloader_val, Total_Samples//BATCH_SIZE):
		images, targets, HWs, frontiers = batch['input'], batch['output'], batch['shape'], batch['frontiers']
		original_targets = batch['original_target']
		#print('images = {}'.format(images))
		#print('targets = {}'.format(targets))
		images, targets = images.cuda(), targets.cuda()

		outputs = model(images)

		images = images.cpu().numpy()
		targets = targets.cpu().numpy()
		outputs = outputs.cpu().numpy()

		for idx in range(BATCH_SIZE):
			print(f'count = {count}')
			occ_map_Mp = images[idx, 0]
			#sem_map_Mp = images[idx, 1]

			target = original_targets[idx]
			output = outputs[idx, 0]

			H, W = HWs[idx]
			frons = frontiers[idx]

			occ_map_Mp = cv2.resize(occ_map_Mp, (W, H), interpolation=cv2.INTER_NEAREST)
			#sem_map_Mp = cv2.resize(sem_map_Mp, (W, H), interpolation=cv2.INTER_NEAREST)

			output = cv2.resize(output, (W, H), interpolation=cv2.INTER_NEAREST)

			#color_sem_map_Mp = apply_color_to_map(sem_map_Mp)
			
			'''
			fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
			ax[0][0].imshow(occ_map_Mp, cmap='gray')
			ax[0][0].get_xaxis().set_visible(False)
			ax[0][0].get_yaxis().set_visible(False)
			ax[0][0].set_title('input: occupancy_map_Mp')
			ax[0][1].imshow(color_sem_map_Mp)
			ax[0][1].get_xaxis().set_visible(False)
			ax[0][1].get_yaxis().set_visible(False)
			ax[0][1].set_title('input: semantic_map_Mp')
			ax[1][0].imshow(target, vmin=0.0, vmax=1.0)
			ax[1][0].get_xaxis().set_visible(False)
			ax[1][0].get_yaxis().set_visible(False)
			ax[1][0].set_title('label: U_a')
			ax[1][1].imshow(output, vmin=0.0, vmax=1.0)
			ax[1][1].get_xaxis().set_visible(False)
			ax[1][1].get_yaxis().set_visible(False)
			ax[1][1].set_title('predict: U_a')

			fig.tight_layout()
			#plt.show()
			fig.savefig(f'{cfg.PRED.PARTIAL_MAP.SAVED_FOLDER}/img_{count}.jpg')
			plt.close()
			'''

			mask_zero = (target == 0)
			output[mask_zero] = 0

			fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 20), dpi=100)
			ax[0][0].imshow(occ_map_Mp, cmap='gray')
			ax[0][0].get_xaxis().set_visible(False)
			ax[0][0].get_yaxis().set_visible(False)
			ax[0][0].set_title('input: occupancy_map_Mp')
			'''
			ax[0][1].imshow(color_sem_map_Mp)
			ax[0][1].get_xaxis().set_visible(False)
			ax[0][1].get_yaxis().set_visible(False)
			ax[0][1].set_title('input: semantic_map_Mp')
			'''
			ax[1][0].imshow(target, vmin=0.0, vmax=1.0)
			ax[1][0].get_xaxis().set_visible(False)
			ax[1][0].get_yaxis().set_visible(False)
			ax[1][0].set_title('label: U_a')
			ax[1][1].imshow(output, vmin=0.0, vmax=1.0)
			ax[1][1].get_xaxis().set_visible(False)
			ax[1][1].get_yaxis().set_visible(False)
			ax[1][1].set_title('predict: U_a')

			#==================== compare each frontier prediction ===============
			for fron in frons:
				points = fron.points.transpose()
				centroid_y, centroid_x = fron.centroid

				vals = target[points[:, 0], points[:, 1]]
				mask_points = points[vals > 0]

				if mask_points.shape[0] > 0:

					target_val = np.mean(target[mask_points[:, 0], mask_points[:, 1]])
					ax[1][0].text(centroid_x, centroid_y, f'{target_val:.2f}', fontsize=20, color='white')

					output_val = np.mean(output[mask_points[:, 0], mask_points[:, 1]])
					ax[1][1].text(centroid_x, centroid_y, f'{output_val:.2f}', fontsize=20, color='white')

			fig.tight_layout()
			#plt.show()
			fig.savefig(f'{cfg.PRED.PARTIAL_MAP.SAVED_FOLDER}/img_{count}_zeroout_occmap_only.jpg')
			plt.close()

			count += 1
			#assert 1==2


