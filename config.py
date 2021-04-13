###################################################################################
# Author: Ricardo Pereira
# Date: 29-03-2021
# Last Modified data: XX-XX-XXXX
# Abstract: SiamRPN: Training and Tracking config file
# Adapted from arbitularov (https://github.com/arbitularov/SiamRPN-PyTorch)
###################################################################################

import numpy as np 

class Config(object):
	# ------------------------- Config for SiamRPN training ------------------------- #
	epoches 			= 200
	train_epoch_size 	= 1000
	val_epoch_size 		= 100

	train_batch_size 	= 8
	valid_batch_size 	= 1
	train_num_workers	= 16
	valid_num_workers	= 16

	start_lr 			= 1e-3
	end_lr				= 1e-5
	warm_lr 			= 1e-3
	warm_scale 			= warm_lr/start_lr
	lr 					= np.logspace(np.log10(start_lr), np.log10(end_lr), num=epoches)
	gamma 				= np.logspace(np.log10(start_lr), np.log10(end_lr), num=epoches)[1] / \
						  np.logspace(np.log10(start_lr), np.log10(end_lr), num=epoches)[0]
	momentum 			= 0.9
	weight_decay 		= 0.0005


	# ------------------------- Config for SiamRPN data ------------------------- #
	template_img_size 	= 127
	detection_img_size 	= 255
	anchor_scales 		= np.array([8,])
	anchor_ratios 		= np.array([0.33, 0.5, 1, 2, 3])
	anchor_num 			= len(anchor_scales) * len(anchor_ratios) # 5
	total_stride		= 12 #8
	score_size 			= 17 
	size 				= anchor_num * score_size * score_size # 1445
	pos_threshold 		= 0.5
	neg_threshold 		= 0.3
	
	out_feature 		= 17
	max_inter 			= 80 
	anchor_base_size	= 8

	context 			= 0.5
	eps 				= 0.01

	max_translate 		= 12
	scale_resize 		= 0.15
	gray_ratio			= 0.25
	exem_stretch		= False

	num_max_pos			= 16
	num_max_neg			= 48
	lamb 				= 1

	# ------------------------- Config for SiamRPN tracking ------------------------- #
	windowing  			= 'cosine'
	penalty_k			= 0.055
	window_influence 	= 0.42
	lr_box 				= 0.30
	min_scale			= 0.1
	max_scale			= 10



config = Config()

if __name__ == '__main__':
	config = Config()
	print(config.lr[0])