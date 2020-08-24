from __future__ import absolute_import, division, print_function
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
import shutil
import torch
import torch.nn as nn
from collections import OrderedDict
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from torch.utils import data
from zplib.image import colorize
from zplib.curve import spline_geometry
import freeimage
import numpy
from elegant import process_images
from elegant import worm_spline
from elegant import datamodel
import torch

import keypoint_dataloader
import keypoint_annotation_model
import keypoint_training

### Load in Data
def has_pose(timepoint):
    pose = timepoint.annotations.get('pose', None)
    # make sure pose is not None, and center/width tcks are both not None
    return pose is not None and pose[0] is not None and pose[1] is not None
    

def has_keypoints(timepoint):
    keypoints = timepoint.annotations.get('keypoints', None)
    return keypoints is not None and not None in keypoints.values() and not False in [x in keypoints.keys() for x in ['anterior bulb', 'posterior bulb', 'vulva', 'tail']]

exp_root1 = '/mnt/lugia_array/20170919_lin-04_GFP_spe-9/'
exp_root2 = '/mnt/lugia_array/20190408_lin-4_spe-9_20C_pos-1/'
#exp_root2 = '/mnt/9karray/Mosley_Matt/20190408_lin-4_spe-9_20C_pos-1/'
exp_root3 = '/mnt/lugia_array/20190813_lin4gfp_spe9_control/20190813_lin4gfp_spe9_control/'
#exp_root3 = '/mnt/scopearray/Mosley_Matt/glp-1/20190813_lin4gfp_spe9_control'


experiments = [datamodel.Experiment(path) for path in (exp_root1, exp_root2, exp_root3)]

#filter experiments
for experiment in experiments:
    experiment.filter(timepoint_filter=(has_pose, has_keypoints))

train, val, test = datamodel.Timepoints.split_experiments(*experiments, fractions=[0.7, 0.2, 0.1])
print(len(train), len(val), len(test))

#model parameters
sets = ['train', 'val']
scale = [0,1,2,3]      # the number of output layer for U-net
batch_size = 5
cropped_size = [960, 960]
total_epoch_num = 25 # total number of epoch in training
base_lr = 0.0005      # base learning rate/
downscale = 1
image_shape = (960,96)

# cpu or cuda
device ='cpu'
if torch.cuda.is_available(): device='cuda:0'

datasets = {'train': keypoint_dataloader.WormKeypointDataset(train, downscale=downscale, scale=scale, image_size=image_shape),
           'val': keypoint_dataloader.WormKeypointDataset(val, downscale=downscale, scale=scale, image_size=image_shape),
           'test': keypoint_dataloader.WormKeypointDataset(test, downscale=downscale, scale=scale, image_size=image_shape)}

dataloaders = {set_name: DataLoader(datasets[set_name], 
                                    batch_size=batch_size,
                                    shuffle=set_name=='train', num_workers = 4)
               for set_name in sets}
#dataloaders = keypoint_dataloader_maps.generate_dataloader_dict(positions, annotations, downscale=1, transform=transform, batch_size=5)
dataset_sizes = {set_name: len(datasets[set_name]) for set_name in sets}
print(dataset_sizes)

project_name = 'new_api_960x96_cov100'
#save_dir = './'+project_name
save_dir = '/mnt/lugia_array/Laird_Nicolette/deep_learning/keypoint_detection/new_api/'+project_name
if not os.path.exists(save_dir): os.makedirs(save_dir)
log_filename = os.path.join(save_dir, 'train.log')

### Initialize model/loss function
#init model 
initModel = keypoint_annotation_model.WormRegModel(34, scale, pretrained=True)
initModel.to(device)

#loss function
loss = keypoint_training.LossofRegmentation(downscale=downscale, scale=scale, image_shape=image_shape)

optimizer = torch.optim.Adam([{'params': initModel.parameters()}], lr=base_lr)

# Decay LR by a factor of 0.5 every int(total_epoch_num/5) epochs
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(total_epoch_num/10), gamma=0.5)

### Train model
new_start = True
ep_time = 0

#add in the the hyperparameters to the log file
fn = open(log_filename, 'a')
fn.write('\nbase_lr: {}\t scale: {}\t start_epo: {}\t total_epoch_nums: {}\t\n'.format(
            base_lr, scale, ep_time, total_epoch_num))
fn.write('dataset_sizes: {}:{}\t {}:{}\t {}:{}\t'.format('train', len(train), 'val', len(val), 'test', len(test)))
fn.write('work_dir: {}\n'.format(save_dir))
fn.close()

keypoint_training.training_wrapper(dataloaders, dataset_sizes, loss, 
	start_epo=ep_time, base_lr=base_lr, total_epoch_nums=total_epoch_num, work_dir=save_dir, device=device)

