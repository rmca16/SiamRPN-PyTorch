###################################################################################
# Author: Ricardo Pereira
# Date: 29-03-2021
# Last Modified data: 01-04-2021
# Abstract: SiamRPN: training data preparation
# Adapted from arbitularov (https://github.com/arbitularov/SiamRPN-PyTorch)
###################################################################################

import os
import sys
import cv2
import time
import random
import numpy as np


import torch
import torch.nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms, utils

from PIL import Image, ImageOps, ImageStat, ImageDraw
from config import config


class Anchor_Boxes(object):
	def __init__(self):
		self.scales 		= config.anchor_scales		# [8,]
		self.ratios 		= config.anchor_ratios		# [0.33, 0.5, 1, 2, 3]
		self.anchor_num		= config.anchor_num			# 5
		self.base_size		= config.anchor_base_size	# 8
		self.score_size		= config.score_size			# 17
		self.total_stride	= config.total_stride		# 12
		self.anchors 		= self.generate_anchors()

	def generate_anchors(self):
		anchor = np.zeros((self.anchor_num, 4), dtype = np.float32) # shape = (5,4)
		size 	= self.base_size * self.base_size 					 # size = 64
		count 	= 0

		for ratio in self.ratios:
			ws = int(np.sqrt(size / ratio)) # 13, 11, 8, 5, 4
			hs = int(ws * ratio)			# 4, 5, 8, 10, 12
			for scale in self.scales:
				wws = ws * scale 			# 104, 88, 64, 40, 32
				hhs = hs * scale 			# 32, 40, 64, 80, 96
				anchor[count, 0] = 0
				anchor[count, 1] = 0
				anchor[count, 2] = wws
				anchor[count, 3] = hhs
				count += 1
		
		anchor = np.tile(anchor, self.score_size * self.score_size).reshape((-1,4)) # (1445, 4)
		ori 	= 25
		xx, yy = np.meshgrid([ori + self.total_stride * dx for dx in range(self.score_size)], # (17,17)
							 [ori + self.total_stride * dy for dy in range(self.score_size)]) # (17,17)
		xx, yy = np.tile(xx.flatten(), (self.anchor_num, 1)).flatten(), \
				 np.tile(yy.flatten(), (self.anchor_num, 1)).flatten()
		anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32) # (1445, 4)

		return anchor


	def pos_neg_anchors(self, bbox):
		norm_anchors = self.anchors_normalization(self.anchors, bbox)
		iou 		 = self.compute_IoU(self.anchors, bbox).flatten()

		pos_index = np.where(iou >= config.pos_threshold)[0][:config.num_max_pos]
		neg_index = np.random.choice(np.where(iou  < config.neg_threshold)[0], config.num_max_neg, replace = False)

		label 	  = np.ones_like(iou) * (- 1)
		label[pos_index] = 1
		label[neg_index] = 0

		return norm_anchors, label


	# Paper's Eq. 3
	def anchors_normalization(self, anchors, gt_bbox):
		norm_anchors = np.zeros_like(anchors, dtype = np.float32)
		norm_anchors[:,0] = (gt_bbox[0] - anchors[:,0]) / (anchors[:,2] + 1e-6)
		norm_anchors[:,1] = (gt_bbox[1] - anchors[:,1]) / (anchors[:,3] + 1e-6)
		norm_anchors[:,2] = np.log((gt_bbox[2] + 1e-6) / (anchors[:,2] + 1e-6))
		norm_anchors[:,3] = np.log((gt_bbox[3] + 1e-6) / (anchors[:,3] + 1e-6))

		return norm_anchors




	def compute_IoU(self, anchors, bbox):
		if np.array(bbox).ndim == 1:
			bbox = np.array(bbox)[None, :] # shape = (1, 4)
		else:
			bbox = np.array(bbox)
		gt_bbox = np.tile(bbox.reshape(1,-1), (anchors.shape[0], 1))	# shape = (1445, 4)

		# Transform cx, cy, w, h => (x1,y1) (x2,y2)
		anchor_x1 = anchors[:, 0] - anchors[:, 2] / 2 + 0.5
		anchor_y1 = anchors[:, 1] - anchors[:, 3] / 2 + 0.5
		anchor_x2 = anchors[:, 0] + anchors[:, 2] / 2 - 0.5
		anchor_y2 = anchors[:, 1] + anchors[:, 3] / 2 - 0.5

		gt_x1 = gt_bbox[:, 0] - gt_bbox[:, 2] / 2 + 0.5
		gt_y1 = gt_bbox[:, 1] - gt_bbox[:, 3] / 2 + 0.5
		gt_x2 = gt_bbox[:, 0] + gt_bbox[:, 2] / 2 - 0.5
		gt_y2 = gt_bbox[:, 1] + gt_bbox[:, 3] / 2 - 0.5

		# Edges values
		xmax = np.max([anchor_x1, gt_x1], axis=0)
		ymax = np.max([anchor_y1, gt_y1], axis=0)
		xmin = np.min([anchor_x2, gt_x2], axis=0)
		ymin = np.min([anchor_y2, gt_y2], axis=0)

		# Intersection
		inter_area = np.max([xmin - xmax, np.zeros(xmax.shape)], axis=0) * \
					 np.max([ymin - ymax, np.zeros(ymax.shape)], axis=0)

		# Area of prediction and ground-truth
		area_anchor = (anchor_x2 - anchor_x1) * (anchor_y2 - anchor_y1)
		area_gt 	= (gt_x2 - gt_x1) * (gt_y2 - gt_y1)

		# Intersection over union
		iou = inter_area / (area_anchor + area_gt - inter_area + 1e-6)

		return iou




# Computes and stores the average and current value
class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val 	= 0
		self.avg 	= 0
		self.sum 	= 0
		self.count 	= 0

	def update(self, val, n = 1):
		self.val 	= val
		self.sum 	+= val * n
		self.count 	+= n
		self.avg 	= self.sum / self.count



class TrainDataLoader(Dataset):
	def __init__(self, data_path, check = False):
		self.max_inter 		= config.max_inter
		self.data_path 		= data_path
		self.ret 	   		= {}
		self.count 			= 0
		self.tmp_dir		= 'tmp/visualization'
		self.check			= check
		self.gen_anchors 	= Anchor_Boxes()
		self.anchors 		= self.gen_anchors.anchors
		self.ret['anchors'] = self.anchors
		self.sub_class_dir 	= [sub_class_dir for sub_class_dir in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, sub_class_dir))]

		if not os.path.isdir(self.tmp_dir):
			os.makedirs(self.tmp_dir)


	# Function to pick template and detection images as well their GT
	def VOT_pick_img_pairs(self, index_of_subclass):
		assert index_of_subclass < len(self.sub_class_dir), 'index_of_subclass should less than total classes'
		
		# ------------- Images Path ------------- #
		sub_class_dir_basename 	= self.sub_class_dir[index_of_subclass] # Gymnastics
		sub_class_dir_path 		= os.path.join(self.data_path, sub_class_dir_basename) # ..\Gymnastics
		sub_class_img_name 		= [img_name for img_name in os.listdir(sub_class_dir_path) if not img_name.find('.jpg') == -1]
		sub_class_img_name 		= sorted(sub_class_img_name) # 000001.jpg ...
		sub_class_img_num  		= len(sub_class_img_name)  # 207
		sub_class_gt_name  		= 'groundtruth.txt'

		status = True
		while status:
			if self.max_inter >= sub_class_img_num-1:
				self.max_inter = sub_class_img_num//2

			#template_index = np.clip(random.choice(range(0, max(1, sub_class_img_num - self.max_inter))), 0, sub_class_img_num-1)
			#detection_index= np.clip(random.choice(range(1, max(2, self.max_inter))) + template_index, 0, sub_class_img_num-1)
			template_index = 50
			detection_index = 113

			template_img_path 	= os.path.join(sub_class_dir_path, sub_class_img_name[template_index])
			detection_img_path 	= os.path.join(sub_class_dir_path, sub_class_img_name[detection_index])
			gt_path				= os.path.join(sub_class_dir_path, sub_class_gt_name)

		# ------------- Labels ------------- #
			with open(gt_path, 'r') as f:
				gt_lines 	= f.readlines()
			template_gt 	= [abs(int(float(i))) for i in gt_lines[template_index].strip('\n').split(',')[:4]]
			detection_gt 	= [abs(int(float(i))) for i in gt_lines[detection_index].strip('\n').split(',')[:4]]

			if template_gt[2]*template_gt[3]*detection_gt[2]*detection_gt[3] != 0:
				status = False
			else:
				print('Warning: encounter object missing, reinitializing...')

		# ------------- Save Template and Detection info ------------- #
		self.ret['template_img_idx'] 		= template_index
		self.ret['detection_img_idx']		= detection_index
		self.ret['template_img_path']		= template_img_path
		self.ret['detection_img_path']		= detection_img_path
		self.ret['template_target_x1y1wh'] 	= template_gt
		self.ret['detection_target_x1y1wh']	= detection_gt
		template_x1y1wh, detection_x1y1wh 	= template_gt.copy(), detection_gt.copy()
		self.ret['template_target_xywh']	= np.array([template_x1y1wh[0]+template_x1y1wh[2]//2, template_x1y1wh[1]+template_x1y1wh[3]//2, template_x1y1wh[2], template_x1y1wh[3]], np.float32)
		self.ret['detection_target_xywh']	= np.array([detection_x1y1wh[0]+detection_x1y1wh[2]//2, detection_x1y1wh[1]+detection_x1y1wh[3]//2, detection_x1y1wh[2], detection_x1y1wh[3]], np.float32)

		if self.check:
			check_dir_path = os.path.join(self.tmp_dir, '0_check_template_detection_bb')
			if not os.path.exists(check_dir_path):
				os.makedirs(check_dir_path)

			template_img 	= Image.open(self.ret['template_img_path'])
			x,y,w,h 		= self.ret['template_target_xywh'].copy()
			x1,y1,x2,y2 	= int(x-w//2), int(y-h//2), int(x+w//2), int(y+h//2)
			draw 			= ImageDraw.Draw(template_img)
			draw.line([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], width=2, fill='red')
			save_path 		= os.path.join(check_dir_path, 'idx_{:04d}_template_img.jpg'.format(self.count))
			template_img.save(save_path)

			detection_img 	= Image.open(self.ret['detection_img_path'])
			x,y,w,h 		= self.ret['detection_target_xywh'].copy()
			x1,y1,x2,y2 	= int(x-w//2), int(y-h//2), int(x+w//2), int(y+h//2)
			draw 			= ImageDraw.Draw(detection_img)
			draw.line([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], width=2, fill='red')
			save_path 		= os.path.join(check_dir_path, 'idx_{:04d}_detection_img.jpg'.format(self.count))
			detection_img.save(save_path)

	
	def VOT_sub_class_img_pairs(self, img_index):
		# ------------- Images Path ------------- #
		sub_class_dir_basename 	= 'Gymnastics' # Gymnastics
		sub_class_dir_path 		= os.path.join(self.data_path, sub_class_dir_basename) # ..\Gymnastics
		sub_class_img_name 		= [img_name for img_name in os.listdir(sub_class_dir_path) if not img_name.find('.jpg') == -1]
		sub_class_img_name 		= sorted(sub_class_img_name) # 000001.jpg ...
		sub_class_img_num  		= len(sub_class_img_name)  # 207
		sub_class_gt_name  		= 'groundtruth.txt'

		status = True
		while status:

			template_index = 0
			detection_index = img_index

			template_img_path 	= os.path.join(sub_class_dir_path, sub_class_img_name[template_index])
			detection_img_path 	= os.path.join(sub_class_dir_path, sub_class_img_name[detection_index])
			gt_path				= os.path.join(sub_class_dir_path, sub_class_gt_name)

		# ------------- Labels ------------- #
			with open(gt_path, 'r') as f:
				gt_lines 	= f.readlines()
			template_gt 	= [abs(int(float(i))) for i in gt_lines[template_index].strip('\n').split(',')[:4]]
			detection_gt 	= [abs(int(float(i))) for i in gt_lines[detection_index].strip('\n').split(',')[:4]]

			if template_gt[2]*template_gt[3]*detection_gt[2]*detection_gt[3] != 0:
				status = False
			else:
				print('Warning: encounter object missing, reinitializing...')

		# ------------- Save Template and Detection info ------------- #
		self.ret['template_img_idx'] 		= template_index
		self.ret['detection_img_idx']		= detection_index
		self.ret['template_img_path']		= template_img_path
		self.ret['detection_img_path']		= detection_img_path
		self.ret['template_target_x1y1wh'] 	= template_gt
		self.ret['detection_target_x1y1wh']	= detection_gt
		template_x1y1wh, detection_x1y1wh 	= template_gt.copy(), detection_gt.copy()
		self.ret['template_target_xywh']	= np.array([template_x1y1wh[0]+template_x1y1wh[2]//2, template_x1y1wh[1]+template_x1y1wh[3]//2, template_x1y1wh[2], template_x1y1wh[3]], np.float32)
		self.ret['detection_target_xywh']	= np.array([detection_x1y1wh[0]+detection_x1y1wh[2]//2, detection_x1y1wh[1]+detection_x1y1wh[3]//2, detection_x1y1wh[2], detection_x1y1wh[3]], np.float32)

		if self.check:
			check_dir_path = os.path.join(self.tmp_dir, '0_check_template_detection_bb')
			if not os.path.exists(check_dir_path):
				os.makedirs(check_dir_path)

			template_img 	= Image.open(self.ret['template_img_path'])
			x,y,w,h 		= self.ret['template_target_xywh'].copy()
			x1,y1,x2,y2 	= int(x-w//2), int(y-h//2), int(x+w//2), int(y+h//2)
			draw 			= ImageDraw.Draw(template_img)
			draw.line([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], width=2, fill='red')
			save_path 		= os.path.join(check_dir_path, 'idx_{:04d}_template_img.jpg'.format(self.count))
			template_img.save(save_path)

			detection_img 	= Image.open(self.ret['detection_img_path'])
			x,y,w,h 		= self.ret['detection_target_xywh'].copy()
			x1,y1,x2,y2 	= int(x-w//2), int(y-h//2), int(x+w//2), int(y+h//2)
			draw 			= ImageDraw.Draw(detection_img)
			draw.line([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], width=2, fill='red')
			save_path 		= os.path.join(check_dir_path, 'idx_{:04d}_detection_img.jpg'.format(self.count))
			detection_img.save(save_path)


	# Function to pre-process template and detection images
	def imgs_pre_processing (self):

		def window_size(bbox, size_z, size_x, context_amount):
			cx, cy, w, h = bbox

			# Paper's Eqs. 12 and 15
			wc_xz 	= w + context_amount * (w + h) 	# w + p, where p = (w+h)/2
			hc_xz 	= h + context_amount * (w + h)	# h + p, where p = (w+h)/2
			s_z  	= int(np.sqrt(wc_xz * hc_xz))	# s_z = A
			scale_z	= size_z / s_z
			#s_x 	= s_z * size_x / size_z # -> approx 2*A 
			s_x 	= s_z * 2	# 2*A

			return s_z, s_x, scale_z

		# ------------- Template ------------- #
		template_img 		= Image.open(self.ret['template_img_path'])
		template_img 		= np.array(template_img)
		template_img_mean 	= np.mean(template_img, axis=(0, 1))

		s_z, s_x, scale = window_size(self.ret['template_target_xywh'],
			config.template_img_size, config.detection_img_size, config.context)

		template_crop_img, scale_z = self.crop_and_pad(template_img, self.ret['template_target_xywh'],
			config.template_img_size, s_z, 'Template', template_img_mean)

		self.ret['template_crop_img'] = template_crop_img

		if self.check:
			check_dir_path = os.path.join(self.tmp_dir, '1_check_template_detection_bb_in padding')
			if not os.path.exists(check_dir_path):
				os.makedirs(check_dir_path)

			template_img = Image.fromarray(self.ret['template_crop_img'].copy(),'RGB')
			save_path 	 = os.path.join(check_dir_path, 'idx_{:04d}_template_cropped_resized.jpg'.format(self.count))
			template_img.save(save_path)

		# ------------- Detection ------------- #
		detection_img 	 	= Image.open(self.ret['detection_img_path'])
		detection_img 		= np.array(detection_img)
		detection_img_mean	= np.mean(detection_img, axis=(0, 1))
		cx, cy, w, h 		= self.ret['detection_target_xywh']

		detection_crop_img, scale_x = self.crop_and_pad(detection_img, self.ret['detection_target_xywh'],
			config.detection_img_size, s_x, 'Detection', detection_img_mean)

		size_x 	= config.detection_img_size
		w_x 	= w * scale_x
		h_x 	= h * scale_x

		x1, y1 	= int(round((size_x + 1) / 2 - w_x / 2)), int(round((size_x + 1) / 2 - h_x / 2))
		x2, y2 	= int(round((size_x + 1) / 2 + w_x / 2)), int(round((size_x + 1) / 2 + h_x / 2))
		cx 		= int(round(x1 + w_x / 2))
		cy 		= int(round(y1 + h_x / 2))


		self.ret['detection_crop_img'] 			= detection_crop_img
		self.ret['detection_crop_resized_xywh']	= np.array((cx, cy, w_x, h_x), dtype = np.int16)


		if self.check:
			detection_img = Image.fromarray(self.ret['detection_crop_img'].copy(),'RGB')
			save_path 	  = os.path.join(check_dir_path, 'idx_{:04d}_detection_padding_resized.jpg'.format(self.count))
			detection_img.save(save_path)

			x, y, w, h 	= self.ret['detection_crop_resized_xywh'].copy()
			x1,y1,x2,y2 	= int(x-w//2), int(y-h//2), int(x+w//2), int(y+h//2)
			draw 			= ImageDraw.Draw(detection_img)
			draw.line([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], width=2, fill='red')
			save_path 		= os.path.join(check_dir_path, 'idx_{:04d}_detection_padding_resized_bb.jpg'.format(self.count))
			detection_img.save(save_path)


	def crop_and_pad(self, img, bbox, model_sz, original_sz, img_type, img_mean = None):

		def round_up(value):
			return round(value + 1e-6 + 1000) - 1000

		cx, cy, w, h 	= bbox
		img_h, img_w, k = img.shape
		
		xmin = cx - (original_sz - 1) / 2
		xmax = xmin + original_sz - 1
		ymin = cy - (original_sz - 1) / 2
		ymax = ymin + original_sz - 1

		left 	= int(round_up(max(0., -xmin)))
		top  	= int(round_up(max(0., -ymin)))
		right 	= int(round_up(max(0., xmax - img_w + 1)))
		bottom 	= int(round_up(max(0., ymax - img_h + 1)))

		xmin = int(round_up(xmin + left))
		xmax = int(round_up(xmax + left))
		ymin = int(round_up(ymin + top))
		ymax = int(round_up(ymax + top))

		if any([top, bottom, left, right]):
			ret_img = np.zeros((img_h + top + bottom, img_w + left + right, k), np.uint8)
			ret_img[top:top + img_h, left:left + img_w, :] = img
			if top: 
				ret_img[0:top, left:left + img_w, :] = img_mean
			if bottom:
				ret_img[img_h + top:, left:left + img_w, :] = img_mean
			if left:
				ret_img[:, 0:left, :] = img_mean
			if right:
				ret_img[:, img_w + left:, :] = img_mean
			img_patch_original = ret_img[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]
		else:
			img_patch_original = img[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]

		if not np.array_equal(model_sz, original_sz):
			img_patch = cv2.resize(img_patch_original, (model_sz, model_sz))
		else:
			img_patch = img_patch_original

		scale = model_sz / img_patch_original.shape[0]

		return img_patch, scale


	def pick_pos_neg_anchors(self):
		norm_anchors, pos_neg_anchors = self.gen_anchors.pos_neg_anchors(self.ret['detection_crop_resized_xywh'])

		self.ret['norm_anchors'] 	= norm_anchors
		self.ret['pos_neg_anchors']	= pos_neg_anchors

		if self.check:
			check_dir_path = os.path.join(self.tmp_dir, '2_check_anchor_boxes')
			if not os.path.exists(check_dir_path):
				os.makedirs(check_dir_path)


			detection_img = Image.fromarray(self.ret['detection_crop_img'].copy(),'RGB')
			detection_img_all_anchors = detection_img.copy()
			draw 		  = ImageDraw.Draw(detection_img_all_anchors)
			x, y, w, h 	  = self.ret['detection_crop_resized_xywh'].copy()

			# ------------- Draw all generated Anchor Boxes ------------- #
			# Transform anchors cx, cy, w, h => (x1,y1) (x2,y2)
			anchor_x1 = self.anchors[:, 0] - self.anchors[:, 2] / 2 + 0.5
			anchor_y1 = self.anchors[:, 1] - self.anchors[:, 3] / 2 + 0.5
			anchor_x2 = self.anchors[:, 0] + self.anchors[:, 2] / 2 - 0.5
			anchor_y2 = self.anchors[:, 1] + self.anchors[:, 3] / 2 - 0.5

			for idx in range(self.anchors.shape[0]):
				an_x1, an_y1, an_x2, an_y2 = anchor_x1[idx], anchor_y1[idx], anchor_x2[idx], anchor_y2[idx]
				draw.line([(an_x1, an_y1), (an_x2, an_y1), (an_x2, an_y2), (an_x1, an_y2), (an_x1, an_y1)], width=1, fill='blue')

			save_path = os.path.join(check_dir_path, 'idx_{:04d}_detection_all_anchor_boxes.jpg'.format(self.count))
			detection_img_all_anchors.save(save_path)

			# ------------- Draw positive and negative Anchor Boxes ------------- #
			detection_img_pos_neg_anchors = detection_img.copy()
			draw = ImageDraw.Draw(detection_img_pos_neg_anchors)

			anchor_labels = self.ret['pos_neg_anchors']
			pos_index = np.where(anchor_labels == 1)[0]
			neg_index = np.where(anchor_labels == 0)[0]

			for idx, pos_idx in enumerate(pos_index):
				an_x1, an_y1, an_x2, an_y2 = anchor_x1[pos_idx], anchor_y1[pos_idx], anchor_x2[pos_idx], anchor_y2[pos_idx]
				draw.line([(an_x1, an_y1), (an_x2, an_y1), (an_x2, an_y2), (an_x1, an_y2), (an_x1, an_y1)], width=1, fill='green')
			save_path = os.path.join(check_dir_path, 'idx_{:04d}_detection_pos_anchor_boxes.jpg'.format(self.count))
			detection_img_pos_neg_anchors.save(save_path)

			for idx, neg_idx in enumerate(neg_index):
				an_x1, an_y1, an_x2, an_y2 = anchor_x1[neg_idx], anchor_y1[neg_idx], anchor_x2[neg_idx], anchor_y2[neg_idx]
				draw.line([(an_x1, an_y1), (an_x2, an_y1), (an_x2, an_y2), (an_x1, an_y2), (an_x1, an_y1)], width=1, fill='red')
			save_path = os.path.join(check_dir_path, 'idx_{:04d}_detection_pos_neg_anchor_boxes.jpg'.format(self.count))
			detection_img_pos_neg_anchors.save(save_path)


	def transform(self):
		transform 		 = transforms.Compose([transforms.ToTensor()])

		template_tensor  = transform(self.ret['template_crop_img'].copy())
		detection_tensor = transform(self.ret['detection_crop_img'].copy())

		self.ret['template_tensor']  		= template_tensor #shape = [1, 3, 127, 127]
		self.ret['detection_tensor'] 		= detection_tensor
		self.ret['norm_anchors_tensor']	 	= torch.Tensor(self.ret['norm_anchors'])
		self.ret['pos_neg_anchors_tensor']	= torch.Tensor(self.ret['pos_neg_anchors'])


	def __len__(self):
		return(207)
		#return len(self.sub_class_dir)

	def __getitem__(self, index):
		self.VOT_sub_class_img_pairs(index)
		self.imgs_pre_processing()
		self.pick_pos_neg_anchors()
		self.transform()
		self.count += 1
		
		return self.ret['template_tensor'], self.ret['detection_tensor'], self.ret['norm_anchors_tensor'], self.ret['pos_neg_anchors_tensor']



if __name__ == '__main__':
	dataset_path = 'S:\\Datasets\\VOT2013\\Data\\'

	train_loader = TrainDataLoader(dataset_path, check = True)
	index_list 	 = range(train_loader.__len__())
	for i in range(1):
		print('\nImage ' + str(i))
		train_loader.__getitem__(random.choice(index_list))
	


