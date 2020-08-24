##### NOTE: Will need to change the worm_frame_image function later if we want to also do vulva classification

from torch.utils import data
from zplib.image import colorize
from zplib.image import pyramid
from zplib.curve import spline_geometry
from zplib.curve import interpolate
import freeimage
import numpy
import pickle
import torch
from elegant import process_images
from elegant import worm_spline
from elegant import datamodel

import keypoint_annotation_model

def to_tck(widths):
    x = numpy.linspace(0, 1, len(widths))
    smoothing = 0.0625 * len(widths)
    return interpolate.fit_nonparametric_spline(x, widths, smoothing=smoothing)

WIDTH_TRENDS = pickle.load(open('/home/nicolette/.conda/envs/nicolette/lib/python3.7/site-packages/elegant/width_data/width_trends.pickle', 'rb'))
AVG_WIDTHS = numpy.array([numpy.interp(5, WIDTH_TRENDS['ages'], wt) for wt in WIDTH_TRENDS['width_trends']])
AVG_WIDTHS_TCK = to_tck(AVG_WIDTHS)
SCALE = [0,1,2,3]

def has_pose(timepoint):
    pose = timepoint.annotations.get('pose', None)
    # make sure pose is not None, and center/width tcks are both not None
    return pose is not None and pose[0] is not None and pose[1] is not None

def has_keypoints(timepoint):
    keypoints = timepoint.annotations.get('keypoints', None)
    return keypoints is not None and not None in keypoints.values() and not False in [x in keypoints.keys() for x in ['anterior bulb', 'posterior bulb', 'vulva', 'tail']]


def get_metadata(timepoint):
    metadata = timepoint.position.experiment.metadata
    try:
        objective, optocoupler, temp = metadata['objective'], metadata['optocoupler'], metadata['nominal_temperature']
    except KeyError:
        objective = 5
        optocoupler = 1
        temp = 25
    magnification = objective * optocoupler
    return objective, optocoupler, magnification, temp

def preprocess_image(timepoint, downscale):
    img_path = timepoint.image_path('bf')
    lab_frame_image = freeimage.read(img_path)
    lab_frame_image = lab_frame_image.astype(numpy.float32)
    height, width = lab_frame_image.shape[:2]
    objective, optocoupler, magnification, temp = get_metadata(timepoint)

    mode = process_images.get_image_mode(lab_frame_image, optocoupler=optocoupler)
    #### DownSample the image 
    if downscale > 0 and downscale != 1:#and set_name!='train':        
        shrink_image = pyramid.pyr_down(lab_frame_image, downscale=downscale)
        #shrink_image = numpy.clip(shrink_image, 0, 40000)   
    else:
        shrink_image = lab_frame_image

    shrink_image = shrink_image.astype(numpy.float32)

    ## scale the image pixel value into a trainable range
    # map image image intensities in range (100, 2*mode) to range (0, 2)
    bf = colorize.scale(shrink_image, min=100, max=2*mode, output_max=2)
    # now shift range to (-1, 1)
    bf -= 1
    return bf

def get_worm_frame_image(timepoint, downscale=1, image_size=(960, 512)):
    bf = preprocess_image(timepoint, downscale)
    annotations = timepoint.annotations
    center_tck, width_tck = annotations['pose']
    image_shape = (image_size[0]/downscale, image_size[1]/downscale)
    #deal with downscaling
    new_center_tck = (center_tck[0], center_tck[1]/downscale, center_tck[2])
    new_width_tck = (width_tck[0], width_tck[1]/downscale, width_tck[2])
    avg_widths = (AVG_WIDTHS_TCK[0], AVG_WIDTHS_TCK[1]/downscale, AVG_WIDTHS_TCK[2])

    reflect = False
    """if 'keypoints' in annotations and 'vulva' in annotations['keypoints']:
                    x, y = annotations['keypoints']['vulva']
                    reflect = y < 0"""

    image_width, image_height = image_shape
    worm_frame = worm_spline.to_worm_frame(bf, new_center_tck, new_width_tck,
        standard_width=avg_widths, zoom=1, order=1, sample_distance=image_height//2, standard_length=image_width, reflect_centerline=reflect)
    mask = worm_spline.worm_frame_mask(avg_widths, worm_frame.shape)
    worm_frame[mask == 0] = 0
    return worm_frame

def process_reg_output(out, downscale=2):
    #Way to get the keypoint maps and make it into the xy positions
    out_kp_map = out[('Keypoint0',0)][0].cpu().detach().numpy()
    out_kp_map = out_kp_map[0]
    image_shape = out_kp_map.shape
    widths_tck = (AVG_WIDTHS_TCK[0], AVG_WIDTHS_TCK[1]/downscale, AVG_WIDTHS_TCK[2])
    mask = worm_spline.worm_frame_mask(widths_tck, image_shape) #make worm mask
    mask = mask>0
    out_kp_map[~mask] = 0 #since we don't care about things outside of the worm pixels, set everything outside to -1
    #out_kp = numpy.where(out_kp_map == numpy.max(out_kp_map[mask]))
    out_kp = numpy.unravel_index(numpy.argmax(out_kp_map), out_kp_map.shape)

    return out_kp
    #return (out_kp[0][0], out_kp[1][0]) #put it into the tuple form

def renormalize_pred_keypoints(timepoint, pred_keypoints, downscale=2, image_size=(960,512)):
    downscale = downscale
    center_tck, width_tck = timepoint.annotations['pose']
    image_shape = (image_size[0]/downscale, image_size[1]/downscale)
    length = spline_geometry.arc_length(center_tck)
    sample_dist = interpolate.spline_interpolate(width_tck, length).max()+20
    width = int(round(sample_dist*2))
    new_keypoints = {}
    for kp, points in pred_keypoints.items():
        x,y = points
        x_percent = x/image_shape[0]
        new_x = x_percent*length
        if kp is 'vulva':
            vulvax = int(new_x)
            print("vulvax: ", vulvax)
            
            print("x_percent: ", x_percent)
            avg_widths = interpolate.spline_interpolate(width_tck, length)
            if vulvax == len(avg_widths):
                vulvax = vulvax-1
            vulvay = avg_widths[vulvax]
            if y <0:
                new_y = -vulvay
            else:
                new_y = vulvay
        else:
            new_y=0

        new_keypoints[kp] = (new_x, new_y)

    return new_keypoints

def normalize_pred_keypoints(timepoint, pred_keypoints, downscale=2, image_size=(960,512)):
    downscale = downscale
    center_tck, width_tck = timepoint.annotations['pose']
    new_width_tck = (AVG_WIDTHS_TCK[0], AVG_WIDTHS_TCK[1]/downscale, AVG_WIDTHS_TCK[2])
    image_shape = (image_size[0]/downscale, image_size[1]/downscale)
    length = spline_geometry.arc_length(center_tck)
    sample_dist = interpolate.spline_interpolate(width_tck, length).max()+20
    if pred_keypoints is None:
        print("Keypoints do not exist with id: ", pred_id)
        return None
    new_keypoints = {}
    for kp, points in pred_keypoints.items():
        x,y = points
        x_percent = x/length
        new_x = x_percent*image_shape[0]
        new_y = int(image_shape[1]/2)
        if kp == 'vulva':
            vulvax = int(new_x)
            avg_widths = interpolate.spline_interpolate(new_width_tck, image_shape[0])
            if vulvax == len(avg_widths):
                vulvax = vulvax-1
            vulvay = avg_widths[vulvax]
            if y <0:
                new_y = (image_shape[1]/2) - vulvay
            else:
                new_y = (image_shape[1]/2) + vulvay
            

        new_keypoints[kp] = (new_x, new_y)

    return new_keypoints

def predict_timepoint_list(timepoint_list, pred_id = 'pred keypoints test', model_paths= {'ant_pharynx':"./models/ant_pharynx_bestValModel.paramOnly", 'post_pharynx':'./models/post_pharynx_bestValModel.paramOnly', 
        'vulva_class':'./models/vulva_class_bestValModel.paramOnly','vuvla_kp':'./models/vulva_kp_flip_bestValModel.paramOnly', 'tail':'./models/tail_bestValModel.paramOnly'}, 
        downscale=2, image_shape=(960,512)):
    """Convience function to predict all timepoints in a timepoint list
    """
    for timepoint in timepoint_list:
        predict_timepoint(timepoint, pred_id, model_paths, downscale, image_shape)

def predict_experiment(experiment, pred_id = 'pred keypoints test', model_paths= {'ant_pharynx':"./models/ant_pharynx_bestValModel.paramOnly", 'post_pharynx':'./models/post_pharynx_bestValModel.paramOnly', 
        'vulva_class':'./models/vulva_class_bestValModel.paramOnly','vuvla_kp':'./models/vulva_kp_flip_bestValModel.paramOnly', 'tail':'./models/tail_bestValModel.paramOnly'}, 
        downscale=2, image_shape=(960,512)):
    """Convience function to predict all timepoints in an Experiment instance
    """
    for position in experiment.positions.values():
        for timepoint in position.timepoints.values():
            print(position.name, timepoint.name)
            predict_timepoint(timepoint, pred_id, model_paths, downscale, image_shape)

def predict_position(position, pred_id = 'pred keypoints test', model_paths= {'ant_pharynx':"./models/ant_pharynx_bestValModel.paramOnly", 'post_pharynx':'./models/post_pharynx_bestValModel.paramOnly', 
        'vulva_class':'./models/vulva_class_bestValModel.paramOnly','vuvla_kp':'./models/vulva_kp_flip_bestValModel.paramOnly', 'tail':'./models/tail_bestValModel.paramOnly'}, 
        downscale=2, image_shape=(960,512)):
    """Convience function to predict all timepoints in Position instance
    """
    for timepoint in position.timepoints.values():
        predict_timepoint(timepoint, pred_id, model_paths, downscale, image_shape)

def predict_timepoint(timepoint, pred_id = 'pred keypoints test', model_paths= {'ant_pharynx':"./models/ant_pharynx_bestValModel.paramOnly", 'post_pharynx':'./models/post_pharynx_bestValModel.paramOnly', 
        'vulva_class':'./models/vulva_class_bestValModel.paramOnly','vuvla_kp':'./models/vulva_kp_flip_bestValModel.paramOnly', 'tail':'./models/tail_bestValModel.paramOnly'}, 
        downscale=2, image_shape=(960,512)):
    
    #get worm-frame image
    worm_frame_image = get_worm_frame_image(timepoint, downscale, image_shape)
    """if keypoints['vulva'][1] > 0:
                        worm_frame_image = numpy.flip(worm_frame_image, axis=1)"""
    worm_frame_image = numpy.expand_dims(worm_frame_image, axis=0)
    extend_img = numpy.concatenate((worm_frame_image, worm_frame_image, worm_frame_image),axis=0)
    #predict image and renormalize keypoints to the original image size
    keypoints = timepoint.annotations['keypoints']
    pred_keypoints = predict_image(extend_img, keypoints, downscale, model_paths)
    
    keypoint_dict = renormalize_pred_keypoints(timepoint, pred_keypoints, downscale, image_shape)
    timepoint.annotations[pred_id] = keypoint_dict
    return keypoint_dict

def predict_image(image, keypoints, downscale=2, model_paths= {'ant_pharynx':"./models/ant_pharynx_bestValModel.paramOnly", 'post_pharynx':'./models/post_pharynx_bestValModel.paramOnly', 'vulva_class':'./models/vulva_class_bestValModel.paramOnly',
        'vulva_kp':'./models/vulva_kp_flip_bestValModel.paramOnly', 'tail':'./models/tail_bestValModel.paramOnly'}):
    
    keypoint_dict = {}
    #identify dorsal, ventral first
    regModel = keypoint_annotation_model.init_vuvla_class_model()
    regModel.load_state_dict(torch.load(model_paths['vulva_class'], map_location='cpu'))
    regModel.eval()
    tensor_img = torch.tensor(image).unsqueeze(0)
    out= regModel(tensor_img)
    _, vulva_out = torch.max(out, 1)
    vulva_out = vulva_out.item()
    #flip vulva if needed
    if keypoints['vulva'][1] > 0:
        tensor_img = torch.flip(tensor_img, [3])
    #print(tensor_img.size())
    print("true vulva: ", keypoints['vulva'][1], "    vulva out: ", out, vulva_out)

    for kp, model_kp in zip(['anterior bulb', 'posterior bulb', 'vulva', 'tail'], ['ant_pharynx', 'post_pharynx', 'vulva_kp', 'tail']):
        #load model
        #print("Loading model: ", model_paths[model_kp])
        regModel = keypoint_annotation_model.WormRegModel(34, SCALE, pretrained=True)
        regModel.load_state_dict(torch.load(model_paths[model_kp], map_location='cpu'))
        regModel.eval()
        
        out = regModel(tensor_img)
        pred_kp = process_reg_output(out, downscale)
        if kp is 'vulva':
            #want to preserve what side the vulva is on.
            if vulva_out <=0:
                x,y = pred_kp
                pred_kp = (x, -y)
        keypoint_dict[kp] = pred_kp

    return keypoint_dict

def predict_worst_timepoint(timepoint, pred_id = 'worst case keypoints', model_paths= {'ant_pharynx':"./models/ant_pharynx_bestValModel.paramOnly", 'post_pharynx':'./models/post_pharynx_bestValModel.paramOnly', 
        'vulva_class':'./models/vulva_class_bestValModel.paramOnly','vuvla_kp':'./models/vulva_kp_flip_bestValModel.paramOnly', 'tail':'./models/tail_bestValModel.paramOnly'}, 
        downscale=2, image_shape=(960,512)):
    
    #get worm-frame image
    worm_frame_image = get_worm_frame_image(timepoint, downscale, image_shape)
    keypoints = timepoint.annotations['keypoints']
    """if keypoints['vulva'][1] < 0:
                    worm_frame_image = numpy.flip(worm_frame_image, axis=1)"""
    #worm_frame_image = numpy.expand_dims(worm_frame_image, axis=0)
    extend_img = numpy.array([worm_frame_image, worm_frame_image, worm_frame_image])
    #extend_img = numpy.concatenate((worm_frame_image, worm_frame_image, worm_frame_image),axis=0)
    #predict image and renormalize keypoints to the original image size
    keypoints = timepoint.annotations['keypoints']
    pred_keypoints = predict_worst_case_image(extend_img, keypoints, downscale, model_paths)
    
    keypoint_dict = renormalize_pred_keypoints(timepoint, pred_keypoints, downscale, image_shape)
    timepoint.annotations[pred_id] = keypoint_dict
    return keypoint_dict

def predict_worst_case_image(image, keypoints, downscale=2, model_paths= {'ant_pharynx':"./models/ant_pharynx_bestValModel.paramOnly", 'post_pharynx':'./models/post_pharynx_bestValModel.paramOnly', 'vulva_class':'./models/vulva_class_bestValModel.paramOnly',
        'vulva_kp':'./models/vulva_kp_flip_bestValModel.paramOnly', 'tail':'./models/tail_bestValModel.paramOnly'}):
    
    keypoint_dict = {}
    tensor_img = torch.tensor(image).unsqueeze(0)
    if keypoints['vulva'][1] < 0:
        tensor_img = torch.flip(tensor_img, [3])
    #identify dorsal, ventral first
    regModel = keypoint_annotation_model.init_vuvla_class_model()
    regModel.load_state_dict(torch.load(model_paths['vulva_class'], map_location='cpu'))
    regModel.eval()
    
    out= regModel(tensor_img)
    _, vulva_out = torch.max(out, 1)
    vulva_out = vulva_out.item()
    
    #print(tensor_img.size())
    print("true vulva: ", keypoints['vulva'][1], "    vulva out: ", out, vulva_out)

    for kp, model_kp in zip(['anterior bulb', 'posterior bulb', 'vulva', 'tail'], ['ant_pharynx', 'post_pharynx', 'vulva_kp', 'tail']):
        #load model
        #print("Loading model: ", model_paths[model_kp])
        regModel = keypoint_annotation_model.WormRegModel(34, SCALE, pretrained=True)
        regModel.load_state_dict(torch.load(model_paths[model_kp], map_location='cpu'))
        regModel.eval()

        out = regModel(tensor_img)
        pred_kp = process_reg_output(out, downscale)
        if kp is 'vulva':
            #want to preserve what side the vulva is on.
            if vulva_out <=0:
                x,y = pred_kp
                pred_kp = (x, -y)
        keypoint_dict[kp] = pred_kp

    return keypoint_dict

def production_predict_image(image, keypoints, downscale=2, model_paths= {'ant_pharynx':"./models/ant_pharynx_bestValModel.paramOnly", 'post_pharynx':'./models/post_pharynx_bestValModel.paramOnly', 'vulva_class':'./models/vulva_class_bestValModel.paramOnly',
        'vulva_kp':'./models/vulva_kp_flip_bestValModel.paramOnly', 'tail':'./models/tail_bestValModel.paramOnly'}):
    
    keypoint_dict = {}
    #identify dorsal, ventral first
    regModel = keypoint_annotation_model.init_vuvla_class_model()
    regModel.load_state_dict(torch.load(model_paths['vulva_class'], map_location='cpu'))
    regModel.eval()
    tensor_img = torch.tensor(image).unsqueeze(0)
    out= regModel(tensor_img)
    _, vulva_out = torch.max(out, 1)
    vulva_out = vulva_out.item()
    #flip vulva if needed
    """if keypoints['vulva'][1] > 0:
                    tensor_img = torch.flip(tensor_img, [3])"""
    #print(tensor_img.size())
    print("true vulva: ", keypoints['vulva'][1], "    vulva out: ", out, vulva_out)

    for kp, model_kp in zip(['anterior bulb', 'posterior bulb', 'vulva', 'tail'], ['ant_pharynx', 'post_pharynx', 'vulva_kp', 'tail']):
        #load model
        #print("Loading model: ", model_paths[model_kp])
        regModel = keypoint_annotation_model.WormRegModel(34, SCALE, pretrained=True)
        regModel.load_state_dict(torch.load(model_paths[model_kp], map_location='cpu'))
        regModel.eval()

        tensor_img = torch.tensor(image).unsqueeze(0)
        out = regModel(tensor_img)
        pred_kp = process_reg_output(out, downscale)
        if kp is 'vulva':
            #want to preserve what side the vulva is on.
            if vulva_out <=0:
                x,y = pred_kp
                pred_kp = (x, -y)
        keypoint_dict[kp] = pred_kp

    return keypoint_dict

def output_prediction_images(image, keypoints, model_paths={'ant_pharynx':"./models/ant_pharynx_bestValModel.paramOnly", 'post_pharynx':'./models/post_pharynx_bestValModel.paramOnly', 'vulva_class':'./models/vulva_class_bestValModel.paramOnly',
        'vulva_kp':'./models/vulva_kp_flip_bestValModel.paramOnly', 'tail':'./models/tail_bestValModel.paramOnly'}):
    keypoint_maps = []
    tensor_img = torch.tensor(image).unsqueeze(0)
    #flip vulva 
    """if keypoints['vulva'][1] > 0:
                    tensor_img = torch.flip(tensor_img, [3])"""
    #print(tensor_img.size())

    for kp, model_kp in zip(['anterior bulb', 'posterior bulb', 'vulva', 'tail'], ['ant_pharynx', 'post_pharynx', 'vulva_kp', 'tail']):
        #load model
        #print("Loading model: ", model_paths[model_kp])
        regModel = keypoint_annotation_model.WormRegModel(34, SCALE, pretrained=True)
        regModel.load_state_dict(torch.load(model_paths[model_kp], map_location='cpu'))
        regModel.eval()

        tensor_img = torch.tensor(image).unsqueeze(0)
        out = regModel(tensor_img)
        out_kp_map = out[('Keypoint0',0)][0].cpu().detach().numpy()
        out_kp_map = out_kp_map[0]
        keypoint_maps.append(out_kp_map)

    return keypoint_maps



######### Analytics functions ##########
def get_tp_accuracy(timepoint, pred_id='pred keypoints test'):
    dist = {}
    gt_kp = timepoint.annotations.get('keypoints', None)
    pose = timepoint.annotations.get('pose', None)
    pred_kp = timepoint.annotations.get(pred_id)
    if gt_kp is None or pred_kp is None or None in gt_kp.values() or None in pred_kp.values():
        print("None found in keypoint")
        return
    elif False in [x in list(gt_kp.keys()) for x in ['anterior bulb','posterior bulb','vulva','tail']]: 
        return
    else:
        for key in pred_kp.keys():
            gtx, gty = gt_kp[key]
            px, py = pred_kp[key]
            if key == 'vulva':
                dist['vulva'] = gtx - px
                dist['vulva class'] = ((gty* py) >= 0) #test sign
            else:
                dist[key] = gtx-px
            

    return dist

def get_accuracy_tplist(timepoint_list, pred_id='pred keypoints test'):
    dist = {'anterior bulb':[], 'posterior bulb':[], 'vulva':[],'vulva class':[], 'tail':[], 'age':[]}
    for timepoint in timepoint_list:
        acc = get_tp_accuracy(timepoint, pred_id)
        age = calculate_tp_age(timepoint)
        #print(acc.keys())
        for kp in acc.keys():
            #print(kp)
            dist[kp] += [acc[kp]]
        dist['age'] += [age]
    return dist

def get_accuracy(experiment, pred_id='pred keypoints test'):
    dist = {'anterior bulb':[], 'posterior bulb':[], 'vulva':[],'vulva class':[], 'tail':[]}

    for worm_name, timepoints in experiment.positions.items():
        for timepoint in timepoints.values():
            gt_kp = timepoint.annotations.get('keypoints')
            pose = timepoint.annotations.get('pose')
            pred_kp = timepoint.annotations.get(pred_id)
            if gt_kp is None or pred_kp is None:
                continue
            if None in gt_kp.values() or None in pred_kp.values():
                continue
            elif False in [x in list(gt_kp.keys()) for x in ['anterior bulb','posterior bulb','vulva','tail']]: 
                continue
            else:
                print(worm_name, timepoint)
                #true_kp = normalize_keypoints(pose, gt_kp, 2)
                for key in pred_kp.keys():
                    print(key)
                    gtx, gty = true_kp[key]
                    px, py = pred_kp[key]
                    if key is 'vulva':
                        dist['vulva']+=[gtx - px]
                        dist['vulva class'] = dist['vulva class']+[((gty* py) >= 0)] #test sign
                        print(key)
                        print("gt: ", gt_kp[key])
                        print("pred: ", pred_kp[key])
                        print("dist: ", str(gtx - px))
                    else:
                        dist[key] = dist[key]+[gtx-px]
            print("dist len: ",len(dist['vulva']))

    error = {key: np.mean(abs(np.array(vals))) for key, vals in dist.items()}
    return dist, error

def calculate_tp_age(timepoint):
    hatch_timestamp = timepoint.position.annotations.get('hatch_timestamp', None)
    if hatch_timestamp is None:
        #assume the first timepoint is the first larval position
        hatch_timestamp = next(iter(timepoint.position.timepoints.values())).annotations['timestamp']
    
    hours = (timepoint.annotations['timestamp']-hatch_timestamp)/3600

    return hours

def get_accuracy_and_age(experiment, pred_id = 'pred keypoints test'):
    dist = {'anterior bulb':[], 'posterior bulb':[], 'vulva':[],'vulva class':[], 'tail':[], 'age': []}    
    for worm_name, positions in experiment.positions.items():
        for timepoint in positions.timepoints.values():
            acc = get_tp_accuracy(timepoint, pred_id)
            age = calculate_tp_age(timepoint)
            #print(acc.keys())
            for kp in acc.keys():
                #print(kp)
                dist[kp] += [acc[kp]]
            dist['age'] += [age]

    return dist

