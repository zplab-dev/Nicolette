import os
from torch.utils import data
from zplib.image import colorize
from zplib.curve import spline_geometry
import freeimage
import numpy
from elegant import process_images
from elegant import worm_spline
from elegant import datamodel
import torch
from datetime import datetime

import model_metrics_new_api
import avg_keypoints

from elegant.torch import dataset
import worm_datasets


save_dir = '/mnt/lugia_array/Laird_Nicolette/deep_learning/keypoint_detection/new_api/'


exp_root1 = '/mnt/lugia_array/20170919_lin-04_GFP_spe-9/'
exp_root2 = '/mnt/lugia_array/20190408_lin-4_spe-9_20C_pos-1/'
#exp_root2 = '/mnt/9karray/Mosley_Matt/20190408_lin-4_spe-9_20C_pos-1/'
exp_root3 = '/mnt/lugia_array/20190813_lin4gfp_spe9_control/20190813_lin4gfp_spe9_control/'
#exp_root3 = '/mnt/scopearray/Mosley_Matt/glp-1/20190813_lin4gfp_spe9_control'

experiments = [datamodel.Experiment(path) for path in (exp_root1, exp_root2, exp_root3)]

def has_pose(timepoint):
    pose = timepoint.annotations.get('pose', None)
    # make sure pose is not None, and center/width tcks are both not None
    return pose is not None and pose[0] is not None and pose[1] is not None
    

def has_keypoints(timepoint):
    keypoints = timepoint.annotations.get('keypoints', None)
    return keypoints is not None and not None in keypoints.values() and not False in [x in keypoints.keys() for x in ['anterior bulb', 'posterior bulb', 'vulva', 'tail']]

#filter experiments
for experiment in experiments:
    experiment.filter(timepoint_filter=(has_pose, has_keypoints))

timepoint_list = datamodel.Timepoints.from_experiments(*experiments)
train, val, test = datamodel.Timepoints.split_experiments(*experiments, fractions=[0.7, 0.2, 0.1])

device ='cpu'
if torch.cuda.is_available(): device='cuda:0'

downscale = 1
image_shape = (960,96)
pred_id = 'pred keypoints 960x96_cov100'

model_path_root = '/mnt/lugia_array/Laird_Nicolette/deep_learning/keypoint_detection/new_api/new_api_960x96_cov100/'
model_paths={'ant_pharynx':model_path_root+"ant_pharynx/bestValModel.paramOnly", 
             'post_pharynx':model_path_root+'post_pharynx/bestValModel.paramOnly', 
             'vulva_class':model_path_root+'Vulva_Classifier/bestValModel.paramOnly',
             'vulva_kp':model_path_root+'vulva_kp/bestValModel.paramOnly', 
             'tail':model_path_root+'tail/bestValModel.paramOnly'}

log_filename = os.path.join(model_path_root,'model_metrics.log')
fn = open(log_filename,'a')
time = datetime.now()
fn.write('---------------- Model metrics run on {} ---------------------\n'.format(time))
fn.write('Model Paths: \n')
for k, p in model_paths.items():
    fn.write('\t{}: {}\n'.format(k,p))
fn.close()

model_metrics_new_api.predict_timepoint_list(test, model_paths=model_paths, pred_id=pred_id, downscale=downscale, image_shape=image_shape)

for timepoint in timepoint_list:
        model_metrics_new_api.predict_worst_timepoint(timepoint, 'worst case keypoints', model_paths, downscale, image_shape)

#output data:
fn = open(log_filename, 'a')
fn.write('Accuracy Metrics: \n')
dist = model_metrics_new_api.get_accuracy_tplist(test, pred_id=pred_id)
for key, acc in dist.items():
    fn.write('{}: {}\n'.format(key,numpy.mean(abs(numpy.array(acc)))))

fn.write("Min/max for each keypoint")
for key, acc in dist.items():
    fn.write('{}: {}, {}\n'.format(key, numpy.min(abs(numpy.array(acc))), numpy.max(abs(numpy.array(acc)))))

worst_dist = model_metrics_new_api.get_accuracy_tplist(test, pred_id='worst case keypoints')
fn.write('Worst Case Metrics: \n')
for key, acc in worst_dist.items():
    fn.write('{}: {}\n'.format(key,numpy.mean(abs(numpy.array(acc)))))

fn.write("Min/max for each keypoint")
for key, acc in worst_dist.items():
    fn.write('{}: {}, {}\n'.format(key, numpy.min(abs(numpy.array(acc))), numpy.max(abs(numpy.array(acc)))))

fn.write('\n\n\n')
fn.close()

#save predictions
for experiment in experiments:
    experiment.write_to_disk()



