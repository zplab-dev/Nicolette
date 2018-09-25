import numpy as np
import freeimage
import pickle

from zplib.image import pyramid
from zplib.curve import interpolate
from zplib.image import colorize
from zplib import pca

from skimage import graph
from scipy import ndimage
from skimage import feature

import scipy.ndimage as ndimage
import scipy.optimize as optimize

from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

import width_finding

mean, pcs, norm_pcs, variances, total_variance,positions, norm_positions = pickle.load(open('/mnt/lugia_array/data/human_masks/warped_splines/pca_stuff.p', 'rb'))
avg_width_positions = pca.pca_reconstruct([0,0,0], pcs[:3], mean)
avg_width_tck = interpolate.fit_nonparametric_spline(np.linspace(0,1, len(avg_width_positions)), avg_width_positions)

def widths_rmsd(image, true_width_tck, ggm_sigma = 3, sig_per = 75, sig_growth_rate = 2, alpha =1, mcp_alpha = 1):
    """Used to optimize the parameters! Wrapper function to generate the widths and then evaluate them
    for the optimization later on
    """
    image_down, traceback = width_finding.find_edges(image, avg_width_tck, ggm_sigma=ggm_sigma, sig_per=sig_per, sig_growth_rate=sig_growth_rate, alpha=alpha, mcp_alpha=mcp_alpha)
    diff = width_finding.evaluate_widths(true_width_tck, traceback, image_down.shape)
    return diff

def rmsd_loss_func(rmsd_list, norm_order=2):
    """Calculate the loss from the training data using
    L-n norm (norm_order determines the order ) 
    """
    return np.linalg.norm(rmsd_list, ord=norm_order)

def get_rmsds(params, input_data):
    """Return the average RMSD/distance metric between all the images.
    This is used in the optimize_params function to pass into the scipy.optimize function

    Parameters:
        params: list of parameter values in the order:
            [ggm sigma, sigmoid percentile, sigmoid growth rate, alpha for the penalties]
        input_data: dictionary of the worm image paths mapped to their true width_tcks and the warped image
            ie. [path to bf image : (warped_image as a np array, true width_tck)]
    """   
    rmsd_list = []
    #input_data = input_data[0]
    #go through all the images and calculate the RMSDs after width finding
    ggm_sigma, sig_per, sig_growth_rate, alpha, mcp_alpha = params

    for img_path, vals in input_data.items():
        warped_image, true_width_tck = vals
        rmsd_list.append(widths_rmsd(warped_image, true_width_tck, ggm_sigma=ggm_sigma, sig_per=sig_per, sig_growth_rate=sig_growth_rate, alpha=alpha, mcp_alpha=mcp_alpha))

    return np.mean(rmsd_list)

def optimize_params(input_data, ranges = ((1,5),(50,75),(0,10),(1,5), (1,5))):
    """Perform the optimization scheme to find the optimal parameters

    Parameters:
        input_data: dictionary of the worm image paths mapped to their true width_tcks and the warped image
            ie. [path to bf image : (warped_image as a np array, true width_tck)]
        init_guess: list of the parameter values that provide the optimizer with an initial guess (same as the x0 parameter
            scipy.optimize.minimize function) the list is in the form:
            [ggm sigma, sigmoid percentile, sigmoid growth rate, alpha for the penalties]
    """
    #x0 = np.array(init_guess)
    #result = optimize.minimize(get_rmsds, x0, args = input_data, method='Nelder_Mead')
    result = optimize.brute(get_rmsds, ranges, args = (input_data,), full_output=True)
    return result