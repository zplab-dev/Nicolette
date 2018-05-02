import freeimage
import pathlib
import numpy as np
import scipy
import pickle
import collections
from functools import partial
from zplib.image import resample
from elegant import worm_data
from zplib.scalar_stats import mcd


def measure_gfp_fluorescence(fluorescent_image, worm_mask, measurement_name=None):
    """Measure specific GFP fluorescent things for a particular GFP image
    and return a dictionary of the measurements

    Parameters:
        flourescent_image: numpy array of the flourescent image
        worm_mask: numpy array of the mask
        measurement_name: string of the descriptor that you want the columns to have
            i.e. if you were measuring gfp in the head then you might put measurement_name = 'head'
            and then in the output your dictionary will have keys such as 'integrated_gfp_head' 
            instead of 'integrated_gfp'. This makes it easier to input into elegant right away and
            not overwrite any other previously made measurements

    Returns:
        gfp_measurements: dictionary of measurements and their values
            NOTE: currently gfp_measurements has set things to look for, but maybe
            in the future we can have this take functions to study.
    """
    worm_pixels = fluorescent_image[worm_mask].copy()

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

    gfp_measurements = {'integrated_gfp' : integrated,
                        'median_gfp' : median, 
                        'percentile95_gfp' : percentile95, 
                        'expressionarea_gfp' : expression_area, 
                        'expressionareafraction_gfp' : expression_area_fraction, 
                        'expressionmean_gfp' : expression_mean, 
                        'highexpressionarea_gfp' : high_expression_area, 
                        'highexpressionareafraction_gfp' : high_expression_area_fraction, 
                        'highexpressionmean_gfp' : high_expression_mean, 
                        'highexpressionintegrated_gfp' : high_expression_integrated}

    
    if measurement_name is not None:
        for key in sorted(gfp_measurements.keys()):
            gfp_measurements[key+"_"+measurement_name] = gfp_measurements.pop(key)

    expression_area_mask = np.zeros(worm_mask.shape).astype('bool')
    high_expression_area_mask = np.zeros(worm_mask.shape).astype('bool')
    percentile95_mask = np.zeros(worm_mask.shape).astype('bool')
    
    expression_area_mask[fluorescent_image > expression_thresh] = True
    expression_area_mask[np.invert(worm_mask)] = False
    high_expression_area_mask[fluorescent_image > high_expression_thresh] = True
    high_expression_area_mask[np.invert(worm_mask)] = False
    percentile95_mask[fluorescent_image > percentile95] = True
    percentile95_mask[np.invert(worm_mask)] = False

    #colored_areas = extractFeatures.color_features([expression_area_mask,high_expression_area_mask,percentile95_mask])  
    return gfp_measurements

def normalize_gfp_image(image, super_vignette, flatfield_image, hot_threshold = 10000):
    """Normalize a GFP image for flatfield
    """

    #apply vignetting
    raw_image = image
    raw_image[np.invert(super_vignette)]=0

    #correct for flatfield
    corrected_image = raw_image*flatfield_image

    #correct for hot pixels
    median_image = scipy.ndimage.filters.median_filter(corrected_image, size = 3)
    difference_image = np.abs(corrected_image.astype('float64') - median_image.astype('float64')).astype('uint16')
    hot_pixels = difference_image > hot_threshold
    median_image_hot_pixels = median_image[hot_pixels]
    corrected_image[hot_pixels] = median_image_hot_pixels

    # Return the actual image.
    return corrected_image  

def warp_image(tcks, image):
    """Warp an image of a worm and return the warped image and a mask

    Parameters:
        tcks: tuple of spine_tck, width_tck; see below for details on the two tcks
            spine_tck: parametric spline tck tuple corresponding to the centerline of the worm
            width_tck: non-parametric spline tck tuple corresponding to the widths of the worm
        image: numpy array corresponding to an image

    Returns:
        Tuple containing (warped, mask)
        warped: numpy array of the warped image of the worm
        mask: numpy array of mask of where the worm is in the image

    """
    spine_tck, width_tck = tcks

    #730 was determined for the number of image samples to take perpendicular to the spine
    #from average length of a worm (as determined from previous tests)
    warped = resample.warp_image_to_standard_width(image, spine_tck, width_tck, width_tck, int(spine_tck[0][-1] // 5))
    #warped = resample.sample_image_along_spline(image, spine_tck, 730)
    mask = resample.make_mask_for_sampled_spline(warped.shape[0], warped.shape[1], width_tck)
    #warped = colorize.scale(warped).astype('uint8')
    #warped[~mask] = 255

    return (warped, mask)

def slice_worms(warp_image, mask, slices):
    """ Measure only a section of the worm.

    Parameters:
        warp_image: numpy array with the image data for the warped worm
        mask: numpy array of the warped worm mask 
            NOTE: warp_image() gives the warp_image and the mask as a tuple
        slice: tuple with the slice you want to take along the length of the worm.
        The tuple will contain percentages of where you want to start and end
            (i.e. if you wanted the first 10% of the worm you would use:
                slice_worms(warp_image, mask, (0,10))

    Returns:
        Tuple of the slice of the warp_image and
        slice of the mask in the form (image_slice, warp_slice)
        
        image_slice: numpy array of the sliced image of the warp_image

        mask: numpy array of the sliced mask
    """

    length = warp_image.shape[0]
    start_per, end_per = slices

    start = start_per/100
    end = end_per/100

    #get the slices from the image
    #Note: to get a percentage along the backbone we need to multiply
    #the length by the start and end percentages
    image_slice = warp_image[int(length*start):int(length*end),]
    mask_slice = mask[int(length*start):int(length*end),]

    return(image_slice, mask_slice)

def measure_slices(flourescent_image, mask, slices={'head':(0,10) , 'anterior': (10,50), 'vulva':(50,60), 'posterior':(60,100)}):
    """TODO make this more generalizable (i.e. instead of hardcoding in the regions
    to analyze, make it so the user can specify where/what to measure either with functions
    or specified slices)

    Gets gfp measurements for many slices of worms. Right now it gets head 
    (1st 10%, anterior half (10%-50%), vulva (50%-60%), posterior half (60%-100%)) and outputs
    it as a dictionary.

    Parameters:
        flourescent_image: numpy array of the flourescent image
        worm_mask: numpy array of the mask
        measurement_name: dictionary of measurement descriptors mapping to the percentages
            to measure for that slice

    Returns:
        slice_measurements: list of measurements

            NOTE: currently slice_measurements has set things to look for, but maybe
            in the future we can have this take functions to study.
    """

    slice_measurements = {}
    for measure, slc in slices.items():
        #print("measuring: ", measure)
        image_slice, mask_slice = slice_worms(flourescent_image, mask, slc)
        gfp = measure_gfp_fluorescence(image_slice, mask_slice, measurement_name=measure)
        slice_measurements.update(gfp)

    return slice_measurements


def measure_worms_experiment(expt_dir, positions, calibration_dir, super_vignette, measurement_name=None, slices=None):
    """Given an experiment, measure GFP flourescence from warped worms for all worms in positions
    and return a dictionary of all the measurements for each timepoint for each worm

    Parameters:
        expt_dir: Experiment where the annotations folder is found
        positions: ordered dictionary of position names (i.e. worm names)
                mapping to ordered dictionaries of timepoint names, each of
                which maps to a list of image paths to load for each timepoint.
                Note: load_data.scan_experiment_dir() provides exactly this.
        calibration_dir: path to the flatfield calibration images for normalization
        super_vignette: path to the super_vignette pickle file Willie generated with his code

    Returns:
        worm_measurements: ordered dictionary of position names (i.e. worm names) mapping
            to a dictionary of gfp measurement names each of which maps to a list of gfp measurement
            values (i.e. {worm_name: {gfp_measure1: [timepoint1_value, timepoint2_value, ...]}})

    """

    expt_dir = pathlib.Path(expt_dir)
    annotation_dir = expt_dir / 'annotations'
    calibration_dir = pathlib.Path(calibration_dir)
    super_vignette = pickle.load(open(super_vignette, 'rb'))

    worm_measurements = collections.OrderedDict()
    #iterate through the worms

    for worm, timepoints in positions.items():
        print("Measuring worm: " + worm)
        annotation_file = list(annotation_dir.glob('*'+worm+'.pickle'))[0]
        timepoint_measurements = measure_worm(timepoints, calibration_dir, annotation_file, super_vignette, measurement_name=measurement_name, slices=slices)
        """_, annotations = pickle.load(open(annotation_file, 'rb'))
                                timepoint_measurements = collections.defaultdict(list)
                                for tp, images in timepoints.items():
                                    #normalize gfp image
                                    raw_image = freeimage.read(images[0])
                                    flatfield_image = freeimage.read(calibration_dir / (tp+" fl_flatfield.tiff"))
                                    corrected_gfp = normalize_gfp_image(raw_image, super_vignette, flatfield_image)
                                    #warp worm to unit worm
                                    tcks = annotations[tp]['pose']
                                    warped_image, mask = warp_image(tcks, corrected_gfp)
                                    #get out gfp measurements
                                    gfp_measurements = measure_gfp_fluorescence(warped_image, mask, measurement_name=measurement_name)
                        
                                    if slices is not None:
                                        slice_measurements = measure_slices(warped_image, mask, slices=slices)
                                        gfp_measurements.update(slice_measurements)
                                    
                                    for measurement, value in gfp_measurements.items():
                                        timepoint_measurements[measurement].append(value)"""

        #populate worm_measurements
        worm_measurements[worm] = timepoint_measurements

    return worm_measurements

def measure_worm(timepoints, calibration_dir, annotation_file, super_vignette, measurement_name=None, slices=None):
    """Measure gfp values for one worm. NOTE: this assumes the worms will be straightened to make the measurements/
    generate the mask

    Parameters:
        timepoints: dictionary of the timepoints mapping to the images to measure
        calibration_dir: path to the calibration folder where all the flatfield images are
        annotation_file: path to the annotation file that goes with the worm (ie. 20170919_lin-04_GFP_spe-9 000.pickle)
        super_vignette: binary array corresponding to the vignette of where the worm/food pad is
    """
    
    _, annotations = pickle.load(open(annotation_file, 'rb'))
    calibration_dir = pathlib.Path(calibration_dir)
    all_timepoints = list(annotations.keys())
    #since we want values for only a few timepoints, make an empty list the length of 
    #all the timepoints and then we will input the values only in the places that make sense
    def empty_timepoints():
        """Since we want an empty numpy array for the timepoint_measurements,
        we gotta do this weird function thingy.
        """
        retVal = np.empty(len(all_timepoints)) * np.nan
        #print(retVal.shape)
        return retVal

    timepoint_measurements = collections.defaultdict(empty_timepoints)
    
    for tp, images in timepoints.items():
            #normalize gfp image
            #print(images[0])
            raw_image = freeimage.read(images[0])
            flatfield_image = freeimage.read(calibration_dir / (tp+" fl_flatfield.tiff"))
            corrected_gfp = normalize_gfp_image(raw_image, super_vignette, flatfield_image)
            #warp worm to unit worm
            tcks = annotations[tp]['pose']
            warped_image, mask = warp_image(tcks, corrected_gfp)
            #get out gfp measurements
            gfp_measurements = measure_gfp_fluorescence(warped_image, mask, measurement_name=measurement_name)

            if slices is not None:
                slice_measurements = measure_slices(warped_image, mask, slices=slices)
                gfp_measurements.update(slice_measurements)
            
            for measurement, value in gfp_measurements.items():
                #need the index of where the timepoitn is
                tp_index = all_timepoints.index(tp)
                timepoint_measurements[measurement][tp_index] = value

    
    return dict(timepoint_measurements)
    



def add_gfp_to_elegant(worms, worm_measurements):
    """Add the gfp measurements from measure_experiment to elegant 
    NOTE: this functions assumes the names of the worms in the Worm
    object is the same as the names in the worm_measurements
    Parameters:
        worms: Worm object from elegant
            NOTE: elegant.read_worms provides this
        worm_measurements: ordered dictionary of position names (i.e. worm names) mapping
            to a dictionary of gfp measurement names each of which maps to a list of gfp measurement
            values (i.e. {worm_name: {gfp_measure1: [timepoint1_value, timepoint2_value, ...]}})
            NOTE: measure_experiment gives this
    """

    for worm in worms:
        gfp_measurements = worm_measurements[worm.name]
        for measurement, values in gfp_measurements.items():
            setattr(worm.td, measurement, values)
