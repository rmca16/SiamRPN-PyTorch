###################################################################################
# Author: Ricardo Pereira
# Date: 05-04-2021
# Last Modified data: 12-04-2021
# Abstract: SiamRPN: Tracking
# Adapted from arbitularov (https://github.com/arbitularov/SiamRPN-PyTorch)
###################################################################################

import numpy as np
import time
import cv2
import os
import sys
import glob
import argparse
from PIL import Image, ImageOps, ImageStat, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data


from data_tracking import *
from SiamRPN_net import SiameseRPN
from config import config

np.set_printoptions(threshold=sys.maxsize)


def SiamRPN_Tracker(cnfg):
	def change(r):
		return np.maximum(r, 1. / r)

	def sz(w, h): # Paper's Eq. 14
		pad = (w + h) * 0.5
		s 	= (w + pad) * (h + pad)
		return np.sqrt(s)

	# ----- Model on GPU ----- #
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	SiamRPN_model = SiameseRPN()
	SiamRPN_model.load_state_dict(torch.load(os.path.join(cnfg.weight_dir,'SiamRPN_model.ckpt')))
	SiamRPN_model.to(device)
	SiamRPN_model.eval()

	print('\nSiamRPN model successfully loaded!')

	# ----- Device Warm-up ----- #
	for i in range(10):
		template_warm_up  = torch.ones((10,3,127,127)).to(device)
		detection_warm_up = torch.ones((10,3,255,255)).to(device)
		SiamRPN_model(template_warm_up, detection_warm_up)
	print('\nSiamRPN Warm-up successfully completed!')

	# ----- Convert image to tensor ----- #
	transform = transforms.Compose([transforms.ToTensor()])


	# ----- Init ----- #
	tracker_data   = VOT_test_img_pairs(cnfg.track_path)
	tracker_loader = TrackerDataLoader()

	# Get Template Img
	z_img, z_img_mean, z_xywh = tracker_data.__getitem__(0)
	z_img, scale_z, s_z, s_x  = tracker_loader.get_template_img(z_img, z_xywh, z_img_mean)

	bbox 		= z_xywh
	target_pos	= np.array([z_xywh[0], z_xywh[1]])
	target_sz	= np.array([z_xywh[2], z_xywh[3]])
	
	# Run network's template branch
	z_img_tensor = transform(z_img).unsqueeze(0).to(device)
	SiamRPN_model.track_init(z_img_tensor)

	# Define window for instance img
	window = np.tile(np.outer(np.hanning(config.score_size), np.hanning(config.score_size))[None, :], [config.anchor_num, 1, 1]).flatten()

	# ----- Update ----- #
	init_time = time.time()
	for i in range(tracker_data.n_imgs):
		# Get Instance Img
		x_img, x_img_mean, x_xywh = tracker_data.__getitem__(i)
		x_img_net, _, _, scale_x  = tracker_loader.get_instance_img(x_img, bbox, s_x, x_img_mean)

		# Run network's detection branch
		x_img_tensor 	= transform(x_img_net).unsqueeze(0).to(device)
		cout, rout 		= SiamRPN_model.track(x_img_tensor) # [1, 2*k, 17, 17]; [1, 4*k, 17, 17]

		cout 	= cout.reshape(-1, 2, config.size).permute(0, 2, 1) # shape = (1, 1445, 2)
		rout 	= rout.reshape(-1, 4, config.size).permute(0, 2, 1) # shape = (1, 1445, 4)
		delta 	= rout[0].cpu().detach().numpy() # shape = (1445, 4)
		
		# Proposal Selection (Paper's Sec. 4.3)
		score_pred 	= F.softmax(cout, dim=2)[0, :, 1].cpu().detach().numpy()

		# --- Proposals Paper Eq. 12 --- #
		box_pred = tracker_loader.gen_anchors.convert_transformed_anchors(tracker_loader.anchors, delta)

		# Check output's positive anchors
		if cnfg.debug:
			if not os.path.exists(os.path.join(cnfg.debug_path, 'pos_anchors')):
				os.makedirs(os.path.join(cnfg.debug_path, 'pos_anchors'))
			x_img_pos = Image.fromarray(x_img_net.copy(),'RGB')
			draw = ImageDraw.Draw(x_img_pos)
			pos_index = np.where(score_pred >= 0.7)[0]
			for idx, pos_idx in enumerate(pos_index):
				an_cx, an_cy, an_w, an_h = box_pred[pos_idx,0], box_pred[pos_idx,1], box_pred[pos_idx,2], box_pred[pos_idx,3]
				an_x1, an_y1, an_x2, an_y2 = an_cx-an_w//2, an_cy-an_h//2, an_cx+an_w//2, an_cy+an_h//2
				draw.line([(an_x1, an_y1), (an_x2, an_y1), (an_x2, an_y2), (an_x1, an_y2), (an_x1, an_y1)], width=1, fill='green')
			save_path = os.path.join(cnfg.debug_path, 'pos_anchors', 'tracking_img_{:04d}.jpg'.format(i))
			x_img_pos.save(save_path)
		

		# Penalty and Window
		s_penalty = change(sz(box_pred[:,2], box_pred[:,3]) / sz(target_sz[0]*scale_x, target_sz[1]*scale_x))
		r_penalty = change((target_sz[0] / target_sz[1]) / (box_pred[:, 2] / box_pred[:, 3]))
		penalty   = np.exp(-(r_penalty * s_penalty - 1.) * config.penalty_k)
		pscore	  = penalty * score_pred
		pscore	  = pscore  * (1 - config.window_influence) + window * config.window_influence
		
		best_pscore_id 	= np.argmax(pscore)
		target 			= box_pred[best_pscore_id, :] / scale_x
		lr 				= penalty[best_pscore_id] * score_pred[best_pscore_id] * config.lr_box

		# Check output's selected bbox
		if cnfg.debug:
			if not os.path.exists(os.path.join(cnfg.debug_path, 'anchor_selection')):
				os.makedirs(os.path.join(cnfg.debug_path, 'anchor_selection'))
			x_img_pos = Image.fromarray(x_img_net.copy(),'RGB')
			draw = ImageDraw.Draw(x_img_pos)
			target_bb = box_pred[best_pscore_id, :]
			an_cx, an_cy, an_w, an_h = target_bb[0], target_bb[1], target_bb[2], target_bb[3]
			an_x1, an_y1, an_x2, an_y2 = an_cx-an_w//2, an_cy-an_h//2, an_cx+an_w//2, an_cy+an_h//2
			draw.line([(an_x1, an_y1), (an_x2, an_y1), (an_x2, an_y2), (an_x1, an_y2), (an_x1, an_y1)], width=1, fill='green')
			save_path = os.path.join(cnfg.debug_path, 'anchor_selection','tracking_img_{:04d}.jpg'.format(i))
			x_img_pos.save(save_path)

		# Update Object's BBox
		res_x = np.clip(target[0] + target_pos[0], 0, x_img.shape[1])
		res_y = np.clip(target[1] + target_pos[1], 0, x_img.shape[0])
		res_w = np.clip(target_sz[0] * (1-lr) + target[2] *  lr, config.min_scale * z_xywh[2],
															  config.max_scale * z_xywh[2])
		res_h = np.clip(target_sz[1] * (1-lr) + target[3] *  lr, config.min_scale * z_xywh[3],
															  config.max_scale * z_xywh[3])

		target_pos 	= np.array([res_x, res_y])
		target_sz 	= np.array([res_w, res_h])
		bbox 		= np.array([np.clip(res_x/2, 0, x_img.shape[1]).astype(np.float64),
								np.clip(res_y/2, 0, x_img.shape[0]).astype(np.float64),
								np.clip(res_w, 10, x_img.shape[1]).astype(np.float64),
								np.clip(res_h, 10, x_img.shape[0]).astype(np.float64)])



		# x_img_pos = Image.fromarray(x_img.copy(),'RGB')
		# draw = ImageDraw.Draw(x_img_pos)
		# an_x1, an_y1, an_x2, an_y2 = bbox[0]-bbox[2]//2, bbox[1]-bbox[3]//2, bbox[0]+bbox[2]//2, bbox[1]+bbox[3]//2
		# draw.line([(an_x1, an_y1), (an_x2, an_y1), (an_x2, an_y2), (an_x1, an_y2), (an_x1, an_y1)], width=2, fill='green')
		# save_path = os.path.join(cnfg.debug_path, 'tracking_img_{:04d}.jpg'.format(i))

	current_time = time.time()
	print("FPS: " + str(1/((current_time-init_time)/tracker_data.n_imgs)))


def create_output_movie(cnfg):
	pos_anchors_imgs_path 		= glob.glob(os.path.join(cnfg.debug_path, 'pos_anchors', '*jpg'))
	selected_anchor_img_path 	= glob.glob(os.path.join(cnfg.debug_path, 'anchor_selection', '*jpg'))

	pos_imgs_array = []; sel_imgs_array = []; combined_imgs_array = []
	for img_id in range(len(pos_anchors_imgs_path)):
		pos_anchor_img = cv2.imread(pos_anchors_imgs_path[img_id])
		sel_anchor_img = cv2.imread(selected_anchor_img_path[img_id])
		combined_imgs  = np.concatenate((pos_anchor_img, sel_anchor_img), axis=1)
		pos_imgs_array.append(pos_anchor_img); sel_imgs_array.append(sel_anchor_img); combined_imgs_array.append(combined_imgs)
	
	pos_imgs_array = np.array(pos_imgs_array)
	sel_imgs_array = np.array(sel_imgs_array)
	combined_imgs_array = np.array(combined_imgs_array)

	pos_sel_size  = (pos_imgs_array[0].shape[1], pos_imgs_array[0].shape[0])
	combined_size = (combined_imgs_array[0].shape[1], combined_imgs_array[0].shape[0])

	pos_anchors_out = cv2.VideoWriter(os.path.join(cnfg.debug_path, 'pos_anchors','tracking.avi'),cv2.VideoWriter_fourcc(*'DIVX'), 15, pos_sel_size)
	sel_anchors_out = cv2.VideoWriter(os.path.join(cnfg.debug_path, 'anchor_selection','tracking.avi'),cv2.VideoWriter_fourcc(*'DIVX'), 15, pos_sel_size)
	com_anchors_out = cv2.VideoWriter(os.path.join(cnfg.debug_path,'tracking.avi'),cv2.VideoWriter_fourcc(*'DIVX'), 15, combined_size)
	
	for i in range(len(pos_imgs_array)):
		pos_anchors_out.write(pos_imgs_array[i])
		sel_anchors_out.write(sel_imgs_array[i])
		com_anchors_out.write(combined_imgs_array[i])

	pos_anchors_out.release()
	sel_anchors_out.release()
	com_anchors_out.release()
	
	



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'PyTorch SiameseRPN Training')
	parser.add_argument('--track_path', default='S:\\Datasets\\VOT2013\\Data\\Gymnastics', metavar='DIR',help='path to dataset')
	parser.add_argument('--weight_dir', default="weights\\", metavar='DIR',help='path to weight')
	parser.add_argument('--debug', default=True, type=bool,  help='whether to debug')
	parser.add_argument('--debug_path', default='tmp/tracking', help='debug tracker path')
	opt = parser.parse_args()

	if not os.path.exists(opt.debug_path):
		os.makedirs(opt.debug_path)

	SiamRPN_Tracker(opt)
	if opt.debug == True:
		create_output_movie(opt)

