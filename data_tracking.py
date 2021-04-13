###################################################################################
# Author: Ricardo Pereira
# Date: 05-04-2021
# Last Modified data: 12-04-2021
# Abstract: SiamRPN: tracking data preparation
# Adapted from arbitularov (https://github.com/arbitularov/SiamRPN-PyTorch)
###################################################################################

import os
import sys
import cv2
import glob
import numpy as np


from PIL import Image
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


	# Paper's Eq. 12
	def convert_transformed_anchors(self, anchors, offset):
		box_cx 	= (anchors[:,0] + offset[:,0] * anchors[:,2]).reshape(-1,1)
		box_cy 	= (anchors[:,1] + offset[:,1] * anchors[:,3]).reshape(-1,1)
		box_w	= (anchors[:,2] * np.exp(offset[:,2])).reshape(-1,1)
		box_h	= (anchors[:,3] * np.exp(offset[:,3])).reshape(-1,1)
		box 	= np.hstack((box_cx, box_cy, box_w, box_h))
		return box



	def pos_neg_anchors(self, bbox):
		norm_anchors = self.anchors_normalization(self.anchors, bbox)
		iou 		 = self.compute_IoU(self.anchors, bbox).flatten()

		pos_index = np.where(iou >= config.pos_threshold)[0][:config.num_max_pos]
		neg_index = np.random.choice(np.where(iou  < config.neg_threshold)[0], config.num_max_neg, replace = False)

		label 	  = np.ones_like(iou) * (- 1)
		label[pos_index] = 1
		label[neg_index] = 0

		return norm_anchors, label



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






class TrackerDataLoader(object):
	def __init__(self, check = False):
		self.size_z			= config.template_img_size
		self.size_x 		= config.detection_img_size
		self.count 			= 0
		self.tmp_dir		= 'tmp/visualization/3_tracking'
		self.check			= check
		self.gen_anchors 	= Anchor_Boxes()
		self.anchors 		= self.gen_anchors.anchors

		if self.check:
			if not os.path.isdir(self.tmp_dir):
				os.makedirs(self.tmp_dir)


	def get_template_img(self, img, bbox, img_mean = None):

		# ---------- def window size (Paper Eq. 15) ---------- #
		cx, cy, w, h = bbox
		wc_xz 	= w + config.context * (w + h) # w + p, where p = (w+h)/2
		hc_xz 	= h + config.context * (w + h)	# h + p, where p = (w+h)/2
		s_z  	= int(np.sqrt(wc_xz * hc_xz))	# s_z = A
		scale_z	= self.size_z / s_z
		s_x 	= s_z * 2	# 2*A

		template_img, _ = self.crop_and_pad(img, cx, cy, self.size_z, s_z, img_mean)

		if self.check:
			template_img = Image.fromarray(template_img.copy(),'RGB')
			save_path 	 = os.path.join(self.tmp_dir, 'tracker_template_input.jpg')
			template_img.save(save_path)

		return template_img, scale_z, s_z, s_x


	def get_instance_img(self, img, template_bbox, s_x, img_mean = None):

		instance_img, scale_x = self.crop_and_pad(img, template_bbox[0], template_bbox[1],
												  self.size_x, s_x, img_mean)

		w_x = template_bbox[2] * scale_x
		h_x = template_bbox[3] * scale_x

		if self.check:
			instance_img = Image.fromarray(instance_img.copy(),'RGB')
			save_path 	 = os.path.join(self.tmp_dir, 'idx_{:04d}_tracker_instance_img_input.jpg'.format(self.count))
			instance_img.save(save_path) 

		return instance_img, w_x, h_x, scale_x


	def crop_and_pad(self, img, cx, cy, model_sz, original_sz, img_mean = None):

		def round_up(value):
			return round(value + 1e-6 + 1000) - 1000

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





class VOT_test_img_pairs(object):
	def __init__(self, data_path, check = False):
		self.data_path 	= data_path
		self.gt_path   	= os.path.join(self.data_path, 'groundtruth.txt')
		self.imgs_path 	= glob.glob(os.path.join(self.data_path, '*jpg'))
		self.n_imgs		= len(self.imgs_path)
		with open(self.gt_path, 'r') as f:
			gt_lines = f.readlines()  # x1 y1 w h
		self.bbox_gt 	= gt_lines

	def __len__(self):
		return self.n_imgs

	def __getitem__(self, index):
		img 	 = Image.open(self.imgs_path[index])
		img 	 = np.array(img)
		img_mean = np.mean(img, axis=(0, 1))
		img_gt	 = [abs(int(float(i))) for i in self.bbox_gt[index].strip('\n').split(',')[:4]]
		img_xywh = np.array([img_gt[0]+img_gt[2]//2, img_gt[1]+img_gt[3]//2, img_gt[2], img_gt[3]], np.float32)

		return img, img_mean, img_xywh





if __name__ == '__main__':

	# ----------- Testing using the VOT 2013 Gymnastics Sub-Dataset ----------- #
	dataset_path = 'S:\\Datasets\\VOT2013\\Data\\Gymnastics'

	# ----------- Pick Template and Detection Imgs ----------- #
	tracking_data 	= VOT_test_img_pairs(dataset_path)
	z_img, z_img_mean, z_xywh = tracking_data.__getitem__(50)
	x_img, x_img_mean, x_xywh = tracking_data.__getitem__(113)

	# ----------- SiamRPN tracker test ----------- #
	tracker_loader = TrackerDataLoader(check = True)

	for i in range(1):
	 	print('\nImage ' + str(i))
	 	template_img_, scale_z, s_z, s_x = tracker_loader.get_template_img(z_img, 
	 									   z_xywh, z_img_mean)
	 	instance_img_, w_x, h_x, scale_x = tracker_loader.get_instance_img(x_img, 
	 									   z_xywh, s_x, z_img_mean)