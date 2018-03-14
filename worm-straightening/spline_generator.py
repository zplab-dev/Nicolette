import pathlib
import freeimage
import numpy as np
import skimage.graph as skg
import PyQt5.Qt as Qt
from skimage.morphology import skeletonize
from skimage.morphology import medial_axis
from scipy import ndimage
from zplib.image import mask
from zplib.image import active_contour
from zplib.curve import interpolate
from zplib.curve import spline_geometry
from zplib.curve import geometry
from zplib.image import resample
from zplib.scalar_stats import mcd
import _pickle as pickle
import straighten_worms
import pandas
import collections
import matplotlib.pyplot as plt

import wormPhysiology.measurePhysiology.extractFeatures as extractFeatures
import wormPhysiology.wormFinding.backgroundSubtraction as backgroundSubtraction
import wormPhysiology.basicOperations.imageOperations as imageOperations


def clean_mask(img):
    '''Clean spurious edges/unwanted things from the masks 
    
    Parameters:
    ------------
    img: array_like (cast to booleans) shape (n,m) 
        Binary image of the worm mask

    Returns:
    -----------
    mask: array_like (cast to booleans) shape (n,m) 
        Binary image of the worm mask without spurious edges
    '''
    #clean up holes in the mask
    img = mask.fill_small_radius_holes(img, 2)
    #dilate/erode to get rid of spurious edges for the mask
    curve_morph = active_contour.CurvatureMorphology(mask=img)
    #TODO: multiple iterations of erode
    curve_morph.erode()
    curve_morph.dilate()
    curve_morph.smooth(iters=2)
    return mask.get_largest_object(curve_morph.mask)


def find_enpoints(skeleton):
    '''Use hit-and-miss tranforms to find the endpoints
    of the skeleton.

    Parameters:
    ------------
    skeleton: array_like (cast to booleans) shape (n,m) 
        Binary image of the skeleton of the worm mask

    Returns:
    -----------
    endpoints: array_like (cast to booleans) shape (n,m)
        Binary image of the endpoints determined by the hit_or_miss transform
    '''

    #structure 1 tells where we want a 1 to be found in the skel
    struct1 = np.array([[0,0,0],[0,1,0],[0,0,0]])
    #struct1 = np.ones((3,3))
    #struct2 tells us where we don't care what the values are
    struct2 = np.array([[1,0,1],[1,0,1],[1,1,1]])
    #struct2 = np.array([[0,1,1],[1,0,1],[1,1,0]])
    struct2_1 = np.array([[1,1,1],[1,0,1],[1,1,0]])
    #struct2 = struct2=np.array([[1,1,1],[0,0,0],[0,0,0]])

    #find all the enpoints
    #need to transpose struct2 to get all the types of endpoints
    #reference: https://homepages.inf.ed.ac.uk/rbf/HIPR2/hitmiss.htm
    ep1 = ndimage.morphology.binary_hit_or_miss(skeleton, structure1=struct1, structure2=struct2)
    ep2 = ndimage.morphology.binary_hit_or_miss(skeleton, structure1=struct1, structure2=struct2.T)
    ep3 = ndimage.morphology.binary_hit_or_miss(skeleton, structure1=struct1, structure2=np.flip(struct2,0))
    ep4 = ndimage.morphology.binary_hit_or_miss(skeleton, structure1=struct1, structure2=np.flip(struct2.T, 1))

    #make sure we don't get zigzags
    ep5 = ndimage.morphology.binary_hit_or_miss(skeleton, structure1=struct1, structure2=struct2_1)
    ep6 = ndimage.morphology.binary_hit_or_miss(skeleton, structure1=struct1, structure2=np.flip(struct2_1,0))
    ep7 = ndimage.morphology.binary_hit_or_miss(skeleton, structure1=struct1, structure2=np.flip(struct2_1,1))
    ep8 = ndimage.morphology.binary_hit_or_miss(skeleton, structure1=struct1, structure2=np.flip(np.flip(struct2_1, 1),0))

    #to get all the endpoints OR the matrices together
    """ print("ep1")
             print(struct2)
             print(np.where(ep1))
             print("ep2")
             print(struct2.T)
             print(np.where(ep2))
             print("ep3")
             print(np.flip(struct2,0))
             print(np.where(ep3))
             print("ep4")
             print(np.flip(struct2.T, 1))
             print(np.where(ep4))
             print("ep5")
             print(struct2_1)
             print(np.where(ep5))
             print("ep6")
             print(np.flip(struct2_1,0))
             print(np.where(ep6))
             print("ep7")
             print(np.flip(struct2_1, 1))
             print(np.where(ep7))
             print("ep8")
             print(np.flip(np.flip(struct2_1, 1),0))
             print(np.where(ep4))
             print("logical or")
             print(np.logical_or.reduce([ep1, ep2, ep3, ep4]).astype(int))"""
    endpoints = np.logical_or.reduce([ep1, ep2, ep3, ep4, ep5, ep6, ep7, ep8])
    return endpoints


def generate_centerline(traceback, shape):
    '''Generate a picture with the centerline defined.
    Makes it easier to view the centerline on RisWidget.
    
     Parameters:
    ------------
    traceback: list of 2-d tuples
        List of indices associated with the centerline, starting with
        one of the endpoints of the centerline and ending with the ending
        index of the centerline. 
    shape: tuple shape (n,m)
        Shape of the image you want the centerline to be generated on.

    Returns:
    -----------
    img: array_like shape (n,m)
        Binary image that indicates where the centerline is
        (1.0 = centerline, 0.0 = not centerline)
    
    '''
    img = np.zeros(shape)
    #x, y = zip(*traceback)
    img[list(np.transpose(traceback))]=1
    return img


def find_centerline(skeleton):
    '''Find the centerline of the worm from the skeleton
    Outputs both the centerline values (traceback and image to view)
    and the spline interpretation of the traceback

    Parameters:
    ------------
    skeleton: array_like (cast to booleans) shape (n,m) 
        Binary image of the skeleton of the worm mask

    Returns:
    -----------
    center: array_like shape (n,m)
        Binary image that indicates where the centerline is
        (1.0 = centerline, 0.0 = not centerline)
    traceback: list of 2-d tuples
        List of indices associated with the centerline, starting with
        one of the endpoints of the centerline and ending with the ending
        index of the centerline. 
    endpoints: array_like (cast to booleans) shape (n,m)
        Binary image of the endpoints determined by the hit_or_miss transform
    '''
    #find enpoints
    endpoints = find_enpoints(skeleton)
    ep_index = list(zip(*np.where(endpoints)))
    #need to change the skeleton to have all zeros be inf
    #to keep the minimal cost function from stepping off the skeleton
    skeleton = skeleton.astype(float)
    skeleton[skeleton==0]=np.inf
    #create mcp object
    mcp = skg.MCP(skeleton)
    #keep track of the longest path/index pair
    traceback = []
    index=(0,0)

    #compute costs for every endpoint pair
    for i in range(0,len(ep_index)-1):
        costs, _=mcp.find_costs([ep_index[i]], ep_index[0:])
        dist=costs[np.where(endpoints)]
        tb = mcp.traceback(ep_index[dist.argmax()])
        #if you find a longer path, then update the longest values
        if len(tb)>len(traceback):
            traceback=tb
            index=(i, dist.argmax())
        

    #print(index)
    #print("length: "+str(len(traceback)))
    #center=[]
    center=generate_centerline(traceback, skeleton.shape)
    #tck = generate_spline(traceback, 15)
    return center,traceback,endpoints


def skel_and_centerline(img):
    '''Generate a skeleton and centerline from an image.
    Main function in the spline-fitting pipeline.

    Parameters:
    ------------
    img: array_like (cast to booleans) shape (n,m) 
        Binary image of the worm mask

    Returns:
    -----------
    mask: array_like (cast to booleans) shape (n,m) 
        Binary image of the worm mask 
    skeleton: array_like (cast to booleans) shape (n,m) 
        Binary image of the skeleton of the worm mask
    centerline: array_like shape (n,m)
        Binary image that indicates where the centerline is
        (1.0 = centerline, 0.0 = not centerline)
    center_dist: ndarray shape(n,m)
        Distance transform from the medial axis transform of the centerline
    medial_axis: ndarray shape(n,m)
        Distance transform from the medial axis transform of the worm mask
    traceback: list of 2-d tuples
        List of indices associated with the centerline, starting with
        one of the endpoints of the centerline and ending with the ending
        index of the centerline. 

    TODO: limit the number of things that are returned to be the most important
    '''
    #need to make sure image only has 0 and 1
    #for the raw mask images, we will need to divide img by 255
    if not np.all(np.in1d(img.flat, (0, 1))):
        img = img/255

    #make mask boolean to work better with ndimage
    img = img>0

    #erode = ndimage.morphology.binary_erosion(img)
    #worm = mask.get_largest_object(erode)
    mask = clean_mask(img)

    #mask = img
    skeleton, med_axis = medial_axis(mask, return_distance=True)
    #med_axis = distance*skeleton
    #skeleton = skeletonize(mask)
    #find centerline
    centerline, traceback, endpoints = find_centerline(skeleton)
    center_dist = centerline*med_axis
    return traceback, med_axis


def center_spline(traceback, distances, smoothing=None):
    '''Generate spline corresponding to the centerline of the worm
    
    Parameters:
    ------------
    traceback: list of 2-d tuples
        List of indices associated with the centerline, starting with
        one of the endpoints of the centerline and ending with the ending
        index of the centerline. 
    distances: ndarray shape(n,m)
        Distance transform from the medial axis transform of the worm mask

    Returns:
    -----------
    tck: parametric spline tuple
        spline tuple (see documentation for zplib.interpolate for more info)
    '''

    #NOTE: we will extrapolate the first/last few pixels to get the full length of the worm,
    #since medial axis transforms/skeltons don't always go to the edge of the mask
    #get the x,y positions for the centerline that can be
    #inputted into the fit spline function
    #NOTE: need to use traceback since order matters for spline creation
    if len(traceback)==0:
        return (0,0,0)

    points = np.array(list(np.transpose(traceback))).T
    #print(points.shape)
    widths = distances[list(np.transpose(traceback))]

    if smoothing is None:
        smoothing = 0.2*len(widths)

    #create splines for the first and last few points
    begin_tck = interpolate.fit_spline(points[:10], smoothing=smoothing, order = 1)
    begin_xys = interpolate.spline_evaluate(begin_tck, np.linspace(-widths[0], 0, int(widths[0]), endpoint=False))
    #print(begin_xys.shape)
    #print(points[-10:])
    end_tck = interpolate.fit_spline(points[-10:], smoothing =smoothing, order = 1)
    tmax = end_tck[0][-1]
    #print(tmax)
    #print(widths[-1])
    #print(tmax+tmax/widths[-1])
    #print(tmax+widths[-1])
    #remove the first point to prevent duplicate points in the end tck
    end_xys = interpolate.spline_evaluate(end_tck, np.linspace(tmax, tmax+widths[-1], int(widths[-1]+1)))[1:]
    #print(end_xys)
    new_points = np.concatenate((begin_xys, points, end_xys))
    #print(new_points)
    #print("new_points: "+str(new_points.shape))
    tck = interpolate.fit_spline(new_points, smoothing=smoothing)
    return tck


def width_spline(traceback, distances, smoothing=None):
    '''Generate a nonparametric spline of the widths along the worm centerline

    Parameters:
    ------------
    traceback: list of 2-d tuples
        List of indices associated with the centerline, starting with
        one of the endpoints of the centerline and ending with the ending
        index of the centerline. 
    distances: ndarray shape(n,m)
        Distance transform from the medial axis transform of the worm mask

    Returns:
    -----------
    tck: nonparametric spline tuple
        spline tuple (see documentation for zplib.interpolate for more info)
    '''
    widths = distances[list(np.transpose(traceback))]
    #print(widths[0])
    #print(widths[-1])
    
    begin_widths = np.linspace(0, widths[0], widths[0], endpoint=False)
    end_widths = np.linspace(widths[-1]-1, 0, widths[-1])
    new_widths = np.concatenate((begin_widths, widths, end_widths))
    x_vals = np.linspace(0,1, len(new_widths))
    #print(x_vals)
    #print(new_widths.shape)
    #print(new_widths)
   
    if smoothing is None:
        smoothing = 0.2*len(widths)
    
    tck = interpolate.fit_nonparametric_spline(x_vals, new_widths, smoothing=smoothing)
    return tck


def generate_splines(mask_dir, export_dir):
    '''Use a directory of masks to generate splines from to 
    generate splines and save the tck into a pickle file

    Parameters:
    ------------

    Returns:
    -----------
    dictionary of the img files and spline tck's
    '''
    mask_dir = pathlib.Path(mask_dir)
    spline_dict = {}

    export_path = pathlib.Path(export_dir)

    if not export_path.exists():
        export_path.mkdir(parents=False)

    for img_path in list(mask_dir.glob("*mask.png")):
        #load in images
        img = freeimage.read(img_path)
        img=img>0
        #crop image for easier/faster spline finding
        sx, sy=ndimage.find_objects(img)[0]
        x = slice(sx.start,sx.stop)
        y = slice(sy.start,sy.stop)
        crop_img = img[x,y]

        timepoint = img_path.stem.split(' ')[0]

        #generate splines
        traceback, medial_axis = skel_and_centerline(crop_img)
        if len(traceback)>0:
            center_tck = center_spline(traceback, medial_axis)
            width_tck = width_spline(traceback, medial_axis)
            #make spline in the same area as the original bf image
            center_tck[1].T[0]+=(sx.start)
            center_tck[1].T[1]+=(sy.start)
        else:
            center_tck = None
            width_tck = None

        #add to spline_dict
        spline_dict[timepoint] = (center_tck, width_tck)

    #save spline_dict out
    output_name = str(export_dir)+"/"+mask_dir.name + "_tck.p"
    print("Pickling spline_dict and saving to: " + str(output_name))

    pickle.dump(spline_dict, open(output_name, 'wb'))

def generate_splines_for_expt(expt_dir):
    '''Assuming the files/masks and things are in the format that Holly has her worms in
    '''
    #get the experiment directories we need
    expt_dir = pathlib.Path(expt_dir)
    work_dir = list(expt_dir.glob('work_dir'))[0]
    worm_ids = get_worm_ids(expt_dir)

    #create the folder where we will store the spline_tcks
    spline_folder = expt_dir.joinpath('spline_tcks')
    spline_folder.mkdir(exist_ok=True)

    #generate splines for every worm
    for worm in worm_ids:
        print("Generating spline for worm "+worm)
        masks = list(work_dir.glob('*'+worm))[0]
        generate_splines(masks, spline_folder)


def get_worm_ids(expt_dir):
    '''From experiment directory, get all the worm #'s'
    '''
    expt_dir = pathlib.Path(expt_dir)
    subdirectories = list(expt_dir.glob('[0-9]*/'))
    worm_ids = []
    for path in subdirectories:
        worm_ids.append(path.name)

    return worm_ids


def warp_worms_from_expt(expt_dir):
    '''After generating the splines for all the worms, warp the bf/gfp images
    NOTE: assume the files/splines are in the folders generated when using the 
    generate_splines_for_expt function and the bf/gfp images are in the same
    format/folders as Holly's data
    '''
    expt_dir = pathlib.Path(expt_dir)
    spline_folder = list(expt_dir.glob('spline_tcks'))[0]
    warped_images = expt_dir.joinpath('warped_images')
    #make new folder for warped images
    warped_images.mkdir(exist_ok=True)
    worm_ids = get_worm_ids(expt_dir)

    #warp images!
    for worm in worm_ids:
        spline_path = list(spline_folder.glob("*"+worm+"*"))[0]
        img_dir = list(expt_dir.glob(worm+"*"))[0]
        export_dir = warped_images.joinpath(worm)
        export_dir.mkdir(exist_ok=True)

        warp_worms(img_dir, spline_path, expt_dir, export_dir)



def get_mask_image(image_file, matt=False):
    """Easy way to find the image file
    Assumed image_dir is in the format: image_dir/worm_run/timestamp bf.png
    Assume image_file is in the format: image_dir/wormm_run/timestamp hmask.png
    """
    #im_dir = pathlib.Path(image_dir)
    if matt:
        #print(image_file.stem)
        timestamp = image_file.stem.split('_')[-1]
        worm_id = image_file.parent
        bf_img = worm_id.joinpath("position_"+timestamp+".png")
    else:
        timestamp = image_file.stem.split(" ")[0]
        worm_id = image_file.parent
        bf_img = worm_id.joinpath(timestamp+" mask.png")
    return bf_img


def warp_worms(img_dir, spline_path, experiment_dir, export_dir, keyword="*gfp.png"):
    '''Once a spline dict has been created and exported in a pickle
    form, use it to warp worms and save the warped images into the export_dir

    NOTE: Assuming that the img_dir is where the spline_dict images came from
    '''

    #spline_dict = pickle.load(open(spline_pickle, 'rb'))
    img_dir = pathlib.Path(img_dir)
    expt_dir = pathlib.Path(experiment_dir)
    #need to get the calibrations, spline pickle, and super_vignette pickle
    spline_dict = pickle.load(open(spline_path, 'rb'))
    calibration_dir = list(expt_dir.glob("calibrations"))[0]
    super_vignette = list(expt_dir.glob("super_vignette.pickle"))[0]
    export_dir = pathlib.Path(export_dir)

    for timepoint in spline_dict.keys():
        img_path = list(img_dir.glob(timepoint+keyword))[0]
        #timepoint = img_path.stem.split(' ')[0]
        center_tck, width_tck = spline_dict[timepoint]

        #assuming we want a bf_image to check our straightening splines
        #NOTE: assuming the bf and gfp images are in the same directory
        bf_image = straighten_worms.get_bf_image(img_path)
        #mask_image = get_mask_image(img_path)
        corrected_gfp_image = read_corrected_gfp_fluorescence(img_path, calibration_dir, super_vignette)

        gfp_warp_name = export_dir.joinpath(img_path.stem+ " warped.png")
        #export_dir+"/"+img_path.stem+" warped.png"
        bf_warp_name = export_dir.joinpath(bf_image.stem+" warped.png")
        #export_dir+"/"+bf_image.stem+" warped.png"
        #mask_warp_name = export_dir.joinpath(mask_image.stem+" warped.png")
        #export_dir+"/"+mask_image.stem+" warped.png"

        print(gfp_warp_name)

        #save both gfp and bf warps
        straighten_worms.warp_image(center_tck, width_tck, corrected_gfp_image, gfp_warp_name)
        straighten_worms.warp_image(center_tck, width_tck, freeimage.read(bf_image), bf_warp_name)
        #straighten_worms.warp_image(center_tck, width_tck, freeimage.read(mask_image), mask_warp_name)


def read_corrected_gfp_fluorescence(gfp_img, calibration_directory, super_vignette, hot_threshold = 10000):
    '''
    Correct fluorescence images for flatfield, and re-normalize to make the images more nicely viewable.    
    '''
    # Read in image and apply vignetting.
    gfp_img = pathlib.Path(gfp_img)
    timepoint = gfp_img.name.split(' ')[0]
    super_vignette = pickle.load(open(super_vignette, 'rb'))
    raw_image = freeimage.read(gfp_img)
    raw_image[np.invert(super_vignette)] = 0
    calibration_directory = pathlib.Path(calibration_directory)

    # Correct for flatfield.
    flatfield_path = calibration_directory.joinpath(timepoint+' fl_flatfield.tiff')
    calibration_image = freeimage.read(flatfield_path)
    #corrected_image = raw_image
    corrected_image = raw_image*calibration_image   

    # Correct for hot pixels.
    median_image = ndimage.filters.median_filter(corrected_image, size = 3)
    difference_image = np.abs(corrected_image.astype('float64') - median_image.astype('float64')).astype('uint16')
    hot_pixels = difference_image > hot_threshold
    median_image_hot_pixels = median_image[hot_pixels]
    corrected_image[hot_pixels] = median_image_hot_pixels

    # Return the actual image.
    return corrected_image.astype(np.uint16)  
    


def get_gfp_exp_measurements(gfp_img_dir, spline_pickle, worm_object):
    '''Get the gfp expression measurements from the warped gfp images
    Worm_object is one of Zach's worm objects that store data from the tsv files
    '''

    #create worm mask from the spline_pickle values
    spline_dict = pickle.load(open(spline_pickle, 'rb'))
    gfp_dir = pathlib.Path(gfp_img_dir)

    warp_gfp_dict = {}

    #for each gfp image, measure the flourescence
    #And then output the measurements into the tsv file
    #as new columns
    for timepoint in spline_dict.keys():
        img_path = list(gfp_dir.glob(timepoint+'*gfp warped.png'))[0]
        #mask_image = list(gfp_dir.glob(timepoint+'*mask warped.png'))[0]
        gfp_img = freeimage.read(img_path)
        center_tck, width_tck = spline_dict[timepoint]
        #mask = freeimage.read(mask_image)
        #mask=mask>0
        mask = resample.make_mask_for_sampled_spline(gfp_img.shape[0], gfp_img.shape[1], width_tck)
        
        gfp_data, colored_areas = measure_gfp_fluorescence(gfp_img, mask)
        warp_gfp_dict[timepoint]=gfp_data

    #add gfp data to the worm_object
    features_to_add = collections.defaultdict(list)
    
    #need to make each feature a list of the particular value
    for timepoint in worm_object.td.timepoint:
        for key, value in warp_gfp_dict[timepoint].items():
            features_to_add[key].append(value)

    features_to_add = dict(features_to_add)

    #add the new rows to the worm_object
    for key, value in features_to_add.items():
        setattr(worm_object.td, key, np.array(value))

    return features_to_add


def measure_gfp_fluorescence(fluorescent_image, worm_mask):
    '''Given an image, measure the gfp flourescence
    TODO: Do we need a worm_frame? Figure out how to add columns to the tsv files
    '''
    worm_pixels = fluorescent_image[worm_mask].copy()

    #Get interesting statistics out
    low_px_mean, low_px_std = mcd.robust_mean_std(worm_pixels[worm_pixels < worm_pixels.mean()], 0.5)
    expression_thresh = low_px_mean + 2.5*low_px_std
    high_expression_thresh = low_px_mean + 6*low_px_std
    fluo_px = worm_pixels[worm_pixels > expression_thresh]
    high_fluo_px = worm_pixels[worm_pixels > high_expression_thresh]
    area = worm_mask.sum()
    integrated = worm_pixels.sum()
    median, percentile95 = np.percentile(worm_pixels, [50, 95])
    expression_area = fluo_px.size
    expression_area_fraction = expression_area / area
    expression_mean = fluo_px.mean()
    high_expression_area = high_fluo_px.size
    high_expression_area_fraction = high_expression_area / area
    high_expression_mean = high_fluo_px.mean()
    high_expression_integrated = high_fluo_px.sum()
    expression_mask = (fluorescent_image > expression_thresh) & worm_mask
    high_expression_mask = (fluorescent_image > high_expression_thresh) & worm_mask

    #create dictionary to store the statistics/measurements

    gfp_data = {
        'warped_integrated_gfp': integrated, 
        'warped_median_gfp': median, 
        'warped_percentile95_gfp': percentile95, 
        'warped_expression_area_gfp': expression_area, 
        'warped_expression_areafraction_gfp': expression_area_fraction, 
        'warped_expression_mean_gfp': expression_mean, 
        'warped_high_expression_area_gfp': high_expression_area, 
        'warped_high_expression_area_fraction_gfp': high_expression_area_fraction, 
        'warped_high_expression_mean_gfp': high_expression_mean, 
        'warped_high_expression_integrated_gfp': high_expression_integrated,
        'warped_total_size': np.sum(worm_mask)
        }

    #Generate nice looking pictures to save
    expression_area_mask = np.zeros(worm_mask.shape).astype('bool')
    high_expression_area_mask = np.zeros(worm_mask.shape).astype('bool')
    percentile95_mask = np.zeros(worm_mask.shape).astype('bool')
    
    expression_area_mask[fluorescent_image > expression_thresh] = True
    expression_area_mask[np.invert(worm_mask)] = False
    high_expression_area_mask[fluorescent_image > high_expression_thresh] = True
    high_expression_area_mask[np.invert(worm_mask)] = False
    percentile95_mask[fluorescent_image > percentile95] = True
    percentile95_mask[np.invert(worm_mask)] = False

    colored_areas = extractFeatures.color_features([expression_area_mask,high_expression_area_mask,percentile95_mask])  
    return (gfp_data, colored_areas)


def plot_warped_vs_holly(worm):
    '''Check to see if holly's and the warped worm stuff looks similar
    '''
    feature_dict ={'warped_integrated_gfp': 'integrated_gfp', 
        'warped_median_gfp': 'median_gfp', 
        'warped_percentile95_gfp': 'percentile95_gfp', 
        'warped_expression_area_gfp': 'expressionarea_gfp', 
        'warped_expression_areafraction_gfp': 'expressionareafraction_gfp', 
        'warped_expression_mean_gfp': 'expressionmean_gfp', 
        'warped_high_expression_area_gfp': 'highexpressionarea_gfp', 
        'warped_high_expression_area_fraction_gfp': 'highexpressionareafraction_gfp', 
        'warped_high_expression_mean_gfp': 'highexpressionmean_gfp', 
        'warped_high_expression_integrated_gfp': 'highexpressionintegrated_gfp',
        'warped_total_size': 'total_size'} 

    warped_features = ['warped_integrated_gfp', 'warped_median_gfp', 
        'warped_percentile95_gfp', 'warped_expression_area_gfp', 
        'warped_expression_areafraction_gfp', 
        'warped_expression_mean_gfp', 
        'warped_high_expression_area_gfp', 
        'warped_high_expression_area_fraction_gfp', 
        'warped_high_expression_mean_gfp', 
        'warped_high_expression_integrated_gfp']

    holly_features = ['integrated_gfp', 'median_gfp', 
        'percentile95_gfp', 'expressionarea_gfp', 
        'expressionareafraction_gfp', 
        'expressionmean_gfp', 
        'highexpressionarea_gfp', 
        'highexpressionareafraction_gfp', 
        'highexpressionmean_gfp', 
        'highexpressionintegrated_gfp']

    for warp_feat, holly_feat in feature_dict.items():
        #get the features
        warped_data = worm.get_time_range(warp_feat, min_age=72, max_age=168)
        holly_data = worm.get_time_range(holly_feat, min_age=72, max_age=168)

        plt.figure(1)
        plt.scatter(warped_data[1], holly_data[1])
        plt.xlabel(warp_feat)
        plt.ylabel(holly_feat)
        plt.title(warp_feat+" vs. "+holly_feat)
        plt.show()







        

