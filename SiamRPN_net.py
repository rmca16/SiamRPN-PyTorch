###################################################################################
# Author: Ricardo Pereira
# Date: 29-03-2021
# Last Modified data: 12-04-2021
# Abstract: SiamRPN: Network
# Adapted from arbitularov (https://github.com/arbitularov/SiamRPN-PyTorch)
###################################################################################

import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import alexnet
from torch.autograd import Variable
from torch import nn

from config import config


class SiameseRPN(nn.Module): # 
	def __init__(self):
		super(SiameseRPN, self).__init__()
		self.device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.featureExtract = nn.Sequential(	# Modified AlexNet
			# nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
			nn.Conv2d(3, 64, kernel_size = 11, stride = 2),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(64, 192, kernel_size=5),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(192, 384, kernel_size=3),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 256, kernel_size=3),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=3),
		)

		self.anchor_num 			= config.anchor_num

		# Classification Branch
		self.conv_cls1 = nn.Conv2d(256, 256 * 2 * self.anchor_num, kernel_size=3)
		self.conv_cls2 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
		self.conv_corr_class = nn.Conv2d(256, 2 * self.anchor_num, kernel_size = 4, bias = False)

		# Regression Branch
		self.conv_r1 = nn.Conv2d(256, 256 * 4 * self.anchor_num, kernel_size=3)
		self.conv_r2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
		self.conv_corr_r = nn.Conv2d(256, 4 * self.anchor_num, kernel_size = 4, bias = False)


	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.normal_(m.weight.data, std= 0.0005)
				nn.init.normal_(m.bias.data, std= 0.0005)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()


	def forward(self, template, detection):
		N 					= template.size(0) 	# [batch, 3, 127, 127] -> size(0) = batch_size
		template_feature 	= self.featureExtract(template) 	# [batch, 256, 6, 6]
		detection_feature 	= self.featureExtract(detection)	# [batch, 256, 22, 22]

		# ---------------- Classification Branch ---------------- #
		kernel_score  = self.conv_cls1(template_feature)				# [batch, 256*2*k, 4, 4]
		kernel_score  = kernel_score.view(N, 2 * self.anchor_num, 256, 4, 4) # [batch, 2*k, 256, 4, 4]
		conv_score    = self.conv_cls2(detection_feature)				# [batch, 256, 20, 20]
		pred_score 	  = torch.zeros(N, 2 * self.anchor_num, 17, 17).to(self.device)
		for i in range(N):
			pred_s = F.conv2d(conv_score[i].unsqueeze(0), kernel_score[i])
			pred_score[i,:,:,:] = pred_s.squeeze(0)

		# ---------------- Regression Branch ---------------- #
		kernel_reg 	  = self.conv_r1(template_feature)				# [batch, 256*4*k, 4, 4]
		kernel_reg 	  = kernel_reg.view(N, 4 * self.anchor_num, 256, 4, 4)
		conv_reg	  = self.conv_r2(detection_feature)				# [batch, 256, 20, 20]
		pred_reg 	  = torch.zeros(N, 4 * self.anchor_num, 17, 17).to(self.device)
		for i in range(N):
			pred_r = F.conv2d(conv_reg[i].unsqueeze(0), kernel_reg[i])
			pred_reg[i,:,:,:] = pred_r.squeeze(0)

		return pred_score, pred_reg


	def track_init(self, template):
		N 				 = template.size(0) # [batch, 3, 127, 127] -> size(0) = batch_size
		template_feature = self.featureExtract(template)

		kernel_score = self.conv_cls1(template_feature) 	# [batch, 256*2*k, 4, 4]
		kernel_score = kernel_score.view(N, 2 * self.anchor_num, 256, 4, 4)

		kernel_reg = self.conv_r1(template_feature)	# [batch, 256*4*k, 4, 4]
		kernel_reg = kernel_reg.view(N, 4 * self.anchor_num, 256, 4, 4)

		self.kernel_score 	= kernel_score
		self.kernel_reg  	= kernel_reg


	def track(self, detection):
		N 				  = detection.size(0) # [batch, 3, 255, 255] -> size(0) = batch_size
		detection_feature = self.featureExtract(detection)

		conv_score 	= self.conv_cls2(detection_feature) 				# [batch, 256*4*k, 4, 4]
		pred_score 	= torch.zeros(N, 2 * self.anchor_num, 17, 17).to(self.device)
		for i in range(N):
			pred_s = F.conv2d(conv_score[i].unsqueeze(0), self.kernel_score[i])
			pred_score[i,:,:,:] = pred_s.squeeze(0)


		conv_reg = self.conv_r2(detection_feature)				# [batch, 256, 20, 20]
		pred_reg = torch.zeros(N, 4 * self.anchor_num, 17, 17).to(self.device)
		for i in range(N):
			pred_r = F.conv2d(conv_reg[i].unsqueeze(0), self.kernel_reg[i])
			pred_reg[i,:,:,:] = pred_r.squeeze(0)

		return pred_score, pred_reg



if __name__ == '__main__':

	template  = torch.ones((10,3,127,127))
	detection = torch.ones((10,3,255,255))

	siamRPN_model = SiameseRPN()
	siamRPN_model(template, detection)

	clas, reg = siamRPN_model(template, detection)
	print(clas.shape); print(str(reg.shape) + '\n')

	siamRPN_model.track_init(template)
	clas, reg = siamRPN_model.track(detection)
	print(clas.shape); print(str(reg.shape) + '\n')


