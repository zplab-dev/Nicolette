import freeimage
import pathlib
import numpy as np
import scipy
import pickle
import collections
from zplib.image import resample
from elegant import worm_data
from zplib.scalar_stats import mcd


def measure_gfp_fluorescence(fluorescent_image, worm_mask):
    """Measure specific GFP fluorescent things for a particular GFP image
    and return a dictionary of the measurements

    Parameters:
        flourescent_image:
        worm_mask:

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

    gfp_measurements = {'integrated_gfp_warp' : integrated,
                        'median_gfp_warp' : median, 
                        'percentile95_gfp_warp' : percentile95, 
                        'expressionarea_gfp_warp' : expression_area, 
                        'expressionareafraction_gfp_warp' : expression_area_fraction, 
                        'expressionmean_gfp_warp' : expression_mean, 
                        'highexpressionarea_gfp_warp' : high_expression_area, 
                        'highexpressionareafraction_gfp_warp' : high_expression_area_fraction, 
                        'highexpressionmean_gfp_warp' : high_expression_mean, 
                        'highexpressionintegrated_gfp_warp' : high_expression_integrated}

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
        warped: warped image of the worm
        mask: mask of where the worm is in the image

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


def measure_experiment(expt_dir, positions, calibration_dir, super_vignette):
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
        _, annotations = pickle.load(open(annotation_file, 'rb'))
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
            gfp_measurements = measure_gfp_fluorescence(warped_image, mask)
            
            for measurement, value in gfp_measurements.items():
                timepoint_measurements[measurement].append(value)

        #populate worm_measurements
        worm_measurements[worm] = dict(timepoint_measurements)

    return worm_measurements

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
