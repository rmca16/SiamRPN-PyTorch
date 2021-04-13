###################################################################################
# Author: Ricardo Pereira
# Date: 02-04-2021
# Last Modified data: 09-04-2021
# Abstract: SiamRPN: Training
# Adapted from arbitularov (https://github.com/arbitularov/SiamRPN-PyTorch)
###################################################################################

import numpy as np
import time
import cv2
import os
import sys
import argparse
from PIL import Image, ImageOps, ImageStat, ImageDraw
from shapely.geometry import Polygon
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data
import torchsummary


from data_train import *
from SiamRPN_net import SiameseRPN
from config import config

np.set_printoptions(threshold=sys.maxsize)


def train_siamRPN(cnfg):

	# ----- Model on GPU ----- #
	SiamRPN_model = SiameseRPN()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	SiamRPN_model.to(device)

	# ----- Loss and Optimizer ----- #
	siamRPN_loss = SiamRPN_loss()
	optimizer 	 = optim.SGD(SiamRPN_model.parameters(), 
						  	 lr 			= config.lr[0],
						  	 momentum 		= config.momentum,
						  	 weight_decay 	= config.weight_decay)

	# ----- Data ----- #
	train_data 		= TrainDataLoader(cnfg.train_path, check = cnfg.debug)
	train_loader 	= data.DataLoader(	dataset 	= train_data,
										batch_size 	= config.train_batch_size,
										shuffle 	= True	)

	# ----- Training phase ----- #
	for epoch in range(config.epoches):
		for param_group in optimizer.param_groups:
			param_group['lr'] = config.lr[epoch]
		with tqdm(int(207/config.train_batch_size)) as progbar:
			for idx, t_data in enumerate(train_loader):
				template, detection, anchors, labels = t_data
				template, detection = template.to(device), detection.to(device)
				anchors, labels		= anchors.to(device), labels.to(device)

				cout, rout = SiamRPN_model(template, detection)

				cout = cout.reshape(-1, 2, config.size).permute(0, 2, 1) # shape = (batch, 1445, 2)
				rout = rout.reshape(-1, 4, config.size).permute(0, 2, 1) # shape = (batch, 1445, 4)

				closs  = siamRPN_loss.cross_entropy(	cout, labels )
				rloss  = siamRPN_loss.smooth_L1( rout, anchors, labels)
				tloss  = siamRPN_loss.loss(closs, rloss) 

				if np.isnan(tloss.cpu().item()):
					raise ValueError("\nTraning phase error!")
					sys.exit(0)

				optimizer.zero_grad()
				tloss.backward()
				optimizer.step()

				siamRPN_loss.losses_update(closs, rloss, tloss)

				progbar.set_postfix(closs = '{:05.3f}'.format(siamRPN_loss.c_losses.avg),
                                    rloss = '{:05.3f}'.format(siamRPN_loss.r_losses.avg),
                                    tloss = '{:05.3f}'.format(siamRPN_loss.t_losses.avg),
                                    lr    = '{:.7f}'.format(config.lr[epoch]),
                                    epoch = '{:04d}'.format(epoch),
                                    batch = '{:03d}'.format(idx))

				progbar.update()

	torch.save(SiamRPN_model.state_dict(), os.path.join(cnfg.weight_dir,'SiamRPN_model.ckpt'))


# ------------------ SiamRPN loss ------------------ #
class SiamRPN_loss(nn.Module):
	def __init__(self):
		super(SiamRPN_loss, self).__init__()

		self.c_losses = AverageMeter()
		self.r_losses = AverageMeter()
		self.t_losses = AverageMeter()
		self.device   = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.lamb	  = config.lamb

	def cross_entropy(self, prediction, target):

		c_loss = []

		for batch_id in range(target.shape[0]):

			pos_index = list(np.where(target[batch_id].cpu() == 1)[0])
			neg_index = list(np.where(target[batch_id].cpu() == 0)[0])

			pos_num , neg_num = len(pos_index), len(neg_index)


			if pos_num > 0:
				c_pos_loss = F.cross_entropy(input 		= prediction[batch_id][pos_index],
											 target		= target[batch_id][pos_index].long(),
											 reduction 	= 'none'	)
			else:
				c_pos_loss = torch.FloatTensor([0]).to(self.device)

			c_neg_loss = F.cross_entropy(input 		= prediction[batch_id][neg_index],
										 target		= target[batch_id][neg_index].long(),
										 reduction 	= 'none'	)

			c_batch_loss = (c_pos_loss.mean() + c_neg_loss.mean()) / 2
			c_loss.append(c_batch_loss)

		c_loss = torch.stack(c_loss).mean()
		return c_loss


	def smooth_L1(self, prediction, target, label):

		r_loss = []

		for batch_id in range(target.shape[0]):

			pos_index = list(np.where(label[batch_id].cpu() == 1)[0])
			pos_num   = len(pos_index)

			if pos_num > 0:
				r_loss_ = F.smooth_l1_loss(	input  		= prediction[batch_id][pos_index],
											target 		= target[batch_id][pos_index],
											reduction 	= 'none' )
			else:
				r_loss_ = torch.FloatTensor([0]).to(self.device)[0]

			r_loss.append(r_loss_.mean())
		r_loss = torch.stack(r_loss).mean()
		return r_loss


	def loss(self, c_loss, r_loss):
		
		siamRPN_loss = c_loss + self.lamb * r_loss
		return siamRPN_loss


	def losses_update(self, c_loss, r_loss, t_loss):

		self.c_losses.update(c_loss.cpu().item())
		self.r_losses.update(r_loss.cpu().item())
		self.t_losses.update(t_loss.cpu().item())

# ---------------- End SiamRPN loss ---------------- #




if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'PyTorch SiameseRPN Training')
	parser.add_argument('--train_path', default='S:\\Datasets\\VOT2013\\Data\\', metavar='DIR',help='path to dataset')
	parser.add_argument('--weight_dir', default="weights\\", metavar='DIR',help='path to weight')
	parser.add_argument('--debug', default=False, type=bool,  help='whether to debug')
	opt = parser.parse_args()

	if not os.path.exists(opt.weight_dir):
		os.makedirs(opt.weight_dir)

	train_siamRPN(opt)