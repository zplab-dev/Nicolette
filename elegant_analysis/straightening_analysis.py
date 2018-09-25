import pathlib
import pickle
import freeimage
import collections
import numpy
import scipy
import time
import matplotlib.pyplot as plt

from elegant import load_data
from elegant import worm_spline

plt.style.use(['seaborn-white', 'presentation'])

def lab_to_lab(lab_frame_image, center_tck, width_tck):
    """Generate an image of the full transform (lab -> worm -> lab).
    This is supposed to help us prove that warping doesn't really do anything
    to the resolution of the image
    """
    lab_mask = worm_spline.lab_frame_mask(center_tck, width_tck, lab_frame_image.shape)
    lab_frame_mask = lab_mask>0

    #go lab->worm->lab
    worm_frame_image = worm_spline.to_worm_frame(lab_frame_image, center_tck, width_tck=width_tck)
    worm_to_lab_image = worm_spline.to_lab_frame(worm_frame_image, lab_frame_image.shape, center_tck, width_tck)

    worm_mask = worm_spline.lab_frame_mask(center_tck, width_tck, worm_to_lab_image.shape)
    worm_frame_mask = worm_mask>0

    return (lab_frame_mask, worm_to_lab_image, worm_frame_mask)

def generate_worm_masks(lab_frame_image, center_tck, width_tck):
    #make a lab frame mask
    lab_mask = worm_spline.lab_frame_mask(center_tck, width_tck, lab_frame_image.shape)
    lab_frame_mask = lab_mask>0

    #get worm_frame image/mask
    worm_frame_image = worm_spline.to_worm_frame(lab_frame_image, center_tck, width_tck=width_tck)
    worm_mask = worm_spline.worm_frame_mask(width_tck, worm_frame_image.shape)
    worm_frame_mask = worm_mask>0

    return (lab_frame_mask, worm_frame_image, worm_frame_mask)

def measure_integrated_gfp(lab_frame_image, lab_frame_mask, worm_frame_image, worm_frame_mask):
    """Evaluate how much GFP pixel values are changed between straightened and
    unstraightened worms. Returns the integrated GFP values for both the worm frame 
    and lab frame 
    """
    lab_worm_pixels = lab_frame_image[lab_frame_mask].copy()
    worm_frame_pixels = worm_frame_image[worm_frame_mask].copy()

    return (lab_worm_pixels.sum(), worm_frame_pixels.sum())

def measure_histogram(lab_frame_image, lab_frame_mask, worm_frame_image, worm_frame_mask):
    """Evaluate the histograms between the lab frame and worm frame images to ensure that the distributions
    haven't changed too much. We will use the chi-squared test to measure the differences between the histograms
    """
    lab_worm_pixels = lab_frame_image[lab_frame_mask].copy()
    worm_frame_pixels = worm_frame_image[worm_frame_mask].copy()
    lab_hist, lab_bins = numpy.histogram(lab_worm_pixels)
    worm_hist, worm_bins = numpy.histogram(worm_frame_pixels, bins=lab_bins) 
    
    #do chi-squared test
    return scipy.stats.chisquare(worm_hist, f_exp=lab_hist)   


def measure_worm(timepoints, annotation_file, mask_generation, measurement_func):
    """Measure the integrated GFP values for one worm

    Parameters:
        timepoints: OrderedDict of the timepoints mapping to the images (GFP) to measure
            NOTE: digging into load_data.scan_experiment_dir gives this
            NOTE: the first image in the images will be the one it will take to measure (needs to be the GFP)
        annotatin+file: path to the annotation file that goes with the worm
        mask_generation: A function that produces a lab frame mask, a worm frame image, and a worm frame mask.
            The function's signature must be: mask_generation(lab_frame_image, center_tck, width_tck) (see lab_to_lab 
            and generate_worm_masks for examples)
    """
    _, annotations = pickle.load(open(annotation_file, 'rb'))
    timepoint_measurements = collections.OrderedDict()
    for tp, image in timepoints.items():
        lab_frame_image = freeimage.read(image[0])
        center_tck, width_tck = annotations[tp]['pose']

        lab_frame_mask, worm_frame_image, worm_frame_mask = mask_generation(lab_frame_image, center_tck, width_tck)
        lab_frame_gfp, worm_frame_gfp = measurement_func(lab_frame_image, lab_frame_mask, worm_frame_image, worm_frame_mask)
        timepoint_measurements[tp] = (lab_frame_gfp, worm_frame_gfp)

    return timepoint_measurements

def measure_experiment(positions, annotation_dir, mask_generation, measurement_func):
    """Measure the integrated gfp for lab frame and worm frames of all images in a given experiment

    Parameters:
        positions: OrderedDict of the worm positions with the timepoints:images
        annotation_dir: path to the directory with the annotation
        mask_generation: A function that produces a lab frame mask, a worm frame image, and a worm frame mask.
            The function's signature must be: mask_generation(lab_frame_image, center_tck, width_tck) (see lab_to_lab 
            and generate_worm_masks for examples)
        measurement_func: function that returns what values should be compared. Function signature must be:
            measurement_func(lab_frame_image, lab_frame_mask, worm_frame_image, worm_frame_mask)
    
    Returns:
    """
    annotation_dir = pathlib.Path(annotation_dir)
    gfp_measurements = collections.OrderedDict()
    for worm_name, timepoints in positions.items():
        gfp_measurements[worm_name] = measure_worm(timepoints, annotation_dir/(worm_name+'.pickle'), mask_generation, measurement_func)

    return gfp_measurements

def extract_gfp_measurements(gfp_measurements):
    """Plot the GFP measurements against each other (linear regression)
    
    Parameters:
        gfp_measurements: OrderedDict of the worm positions with the timepoints and their gfp values
    """
    plot_vals = []
    for worm in gfp_measurements.keys():
        plot_vals.extend(list(gfp_measurements[worm].values()))

    return plot_vals
    

def plot(measurements, save_dir, features, days="all"):
    """Function to plot things in case you want to only plot
    a few features

    Parameters:
        measurements: tuple of list of measurements to plot (x,y)
            In general, this is (lifespan, gfp_measurement)
        save_dir: place to save the files
        features: names of the features you are trying to plot,
            inputted in the same way as the measurments (x,y)
    
    Returns:
    """
    save_dir = pathlib.Path(save_dir)
    save_path = save_dir / (features[0]+" vs "+features[1]+" "+days+" dph.png")

    #get the regression line stuff
    pearson,spearman,yp=run_stats(measurements[0],measurements[1])

    plt.scatter(measurements[0], measurements[1])
    plt.plot(measurements[0],yp, c='gray')
    title = features[0]+" vs "+features[1]+" at "+days+" dph"
    plt.style.use('seaborn-white')
    plt.xlabel((features[0]+" (dph)"), fontdict={'size':20,'family':'calibri'})
    plt.ylabel(("Mean "+features[1]), fontdict={'size':20,'family':'calibri'})
    plt.title(title, y=1.05,fontdict={'size':26,'weight':'bold','family':'calibri'})


    if pearson[1]<.00001:
        p="p<.00001"
    else:
        p="p=" + ''+ (str)(round(pearson[1],3))
    if spearman[1]<.00001:
        spearman_p="p<.00001"
    else:
        spearman_p="p=" + '' + (str)(round(spearman[1],3))
                
    ftext='$r^{2}$ = '+(str)(round(pearson[0]**2,3))+" "+p
    gtext=r'$\rho$'+" = "+(str)(round(spearman[0],3))+" "+spearman_p
    
    plt.figtext(.15,.85,ftext,fontsize=20,ha='left')
    plt.figtext(.15,.8,gtext,fontsize=20,ha='left')
 
    plt.gcf()
    plt.savefig(save_path.as_posix())
    plt.show(block=False)
    time.sleep(1)
    plt.close()
    plt.gcf().clf

def run_stats(x_list,y_list):
    """Get the pearson, spearman, and polyfit coorelations from
    the data.
    """
    pearson=numpy.asarray(scipy.stats.pearsonr(x_list, y_list))
    spearman=numpy.asarray(scipy.stats.spearmanr(x_list, y_list))
    (m,b) = numpy.polyfit(x_list, y_list, 1)
    yp=numpy.polyval([m,b], x_list)
    
    return (pearson,spearman, yp)

