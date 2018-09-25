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

from elegant import worm_spline
import celiagg
import freeimage
from ris_widget import histogram_mask

mean, pcs, norm_pcs, variances, total_variance,positions, norm_positions = pickle.load(open('/mnt/lugia_array/data/human_masks/warped_splines/pca_stuff.p', 'rb'))
avg_width_positions = pca.pca_reconstruct([0,0,0], pcs[:3], mean)
avg_width_tck = interpolate.fit_nonparametric_spline(np.linspace(0,1, len(avg_width_positions)), avg_width_positions)

def optimize_params(image, true_width_tck, ggm_sigma = 3, sig_per = 75, sig_growth_rate = 2, alpha =1):
    """Used to optimize the parameters! Wrapper function to generate the widths and then evaluate them
    for the optimization later on
    """
    image_down, traceback, width_image, new_costs, y_grad, sig = width_finding(image, ggm_sigma=ggm_sigma, sig_per=sig_per, sig_growth_rate=sig_growth_rate, alpha=alpha)
    diff = evaluate_widths(true_width_tck, traceback, image_down.shape)
    return diff

def evaluate_widths(true_width_tck, traceback, shape):
    """with the found widths evaluate how good they are
    compared to the true widths via RMSD
    """
    widths = interpolate.spline_interpolate(true_width_tck, len(traceback))
    #since we are looking at the top part of the image, we need to get the widths
    #to reflect that
    true_widths = widths/2
    #true_widths = shape[1]-widths-1
    #true_width_image = visualize_widths(widths, width_image.shape)
    x,y = np.transpose(traceback)
    diff = np.sqrt(np.sum((true_widths-y)**2)*(1/len(true_widths)))
    #diff = np.linalg.norm(true_widths-widths, ord=None)
    return diff

def find_widths(image, true_width_tck, ggm_sigma=3, sig_per=75, sig_growth_rate=2, alpha =1, mcp_alpha=1):
    """Find the top and bottom widths from the image
    """
    
    top_image = image[:, :int(image.shape[1]/2)]
    top_image_down, top_traceback, top_width_image, top_new_costs, top_y_grad, top_sig = width_finding(top_image, ggm_sigma=ggm_sigma, 
                sig_per=sig_per, sig_growth_rate=sig_growth_rate, alpha=alpha, mcp_alpha=mcp_alpha)
    widths = interpolate.spline_interpolate(true_width_tck, top_image_down.shape[0])
    true_widths = widths/2
    top_true_widths = visualize_widths(true_widths, top_image_down.shape)

    bottom_image = np.flip(image[:,int(image.shape[1]/2):],axis=1)
    bottom_image_down, bottom_traceback, bottom_width_image, bottom_new_costs, bottom_y_grad, bottom_sig = width_finding(bottom_image, ggm_sigma=ggm_sigma, 
                sig_per=sig_per, sig_growth_rate=sig_growth_rate, alpha=alpha)
    
    #put them together!
    full_image_down = np.hstack((top_image_down, np.flip(bottom_image_down, axis=1)))
    full_traceback = np.hstack((top_width_image, np.flip(bottom_width_image, axis=1)))
    full_true_width_image = np.hstack((top_true_widths, np.flip(top_true_widths, axis=1)))
    full_new_costs = np.hstack((top_new_costs, np.flip(bottom_new_costs, axis=1)))
    full_y_grad = np.hstack((top_y_grad, np.flip(bottom_y_grad, axis=1)))
    full_sig = np.hstack((top_sig, np.flip(bottom_sig, axis=1)))

    return (full_image_down, full_traceback, full_true_width_image, full_new_costs, full_y_grad, full_sig)

def sigmoid(gradient, min, mid, max, growth_rate):
    '''Apply the sigmoid function to the gradient.

    Parameters:
        gradient: array of the gradient of the image
        min: lower asymptote of the sigmoid function
        mid: midpoint of the sigmoid function (ie. the point that is halfway between
            the lower and upper asymptotes)
        max: upper asymptote
        growth_rate: growth rate of the sigmoid function
    '''
    #return min+((max-min)/(1+y0*np.exp(-(growth_rate)*(gradient))))
    return min+((max-min)/(1+np.exp(-(growth_rate)*(gradient-mid))))

def find_edges(image, avg_width_tck, ggm_sigma=1, sig_per=61, sig_growth_rate=2, alpha=1, mcp_alpha=1):
    """Find the edges of one side of the worm and return the x,y positions of the new widths
    NOTE: This function assumes that the image is only half of the worm (ie. from the centerline
    to the edges of the worm)

    Parameters:
        image: ndarray of the straightened worm image (typically either top or bottom half)
        avg_width_tck: width spline defining the average distance from the centerline
            to the worm edges (This is taken from the pca things we did earlier)
        ggm_sigma, sig_per, sig_growth_rate, alpha, mcp_alpha: hyperparameters for 
            the edge-detection scheme
    
    Returns:
        route: tuple of x,y positions of the identfied edges
    """

    #down sample the image
    image_down = pyramid.pyr_down(image, downscale=2)

    #get the gradient
    gradient = ndimage.filters.gaussian_gradient_magnitude(image_down, ggm_sigma)
    print (sig_per)
    top_ten = np.percentile(gradient,  sig_per)
    gradient = sigmoid(gradient, gradient.min(), top_ten, gradient.max(), sig_growth_rate)
    gradient = gradient.max()-abs(gradient)

    #penalize finding edges near the centerline or outside of the avg_width_tck
    #since the typical worm is fatter than the centerline and not huge
    #Need to divide by 2 because of the downsampling
    pen_widths = (interpolate.spline_interpolate(avg_width_tck, image_down.shape[0]))
    #pen_widths = pen_widths/2
    distance_matrix = abs(np.subtract.outer(pen_widths, np.arange(0, image_down.shape[1])))
    #distance_matrix = np.flip(abs(np.subtract.outer(pen_widths, np.arange(0, image_down.shape[1]))), 1)
    penalty = alpha*(distance_matrix)
    new_costs = gradient+penalty
    
    #set start and end points for the traceback
    start = (0, int(pen_widths[0]))
    end = (len(pen_widths)-1, int(pen_widths[-1]))
    #start = (0, int((image_down.shape[1]-1)-pen_widths[0]))
    #end = (len(pen_widths)-1, int((image_down.shape[1]-1)-pen_widths[-1]))
    #start = (0,0)
    #end = (len(pen_widths)-1, 0)

    #begin edge detection
    offsets= [(1,-1),(1,0),(1,1)]
    mcp = Smooth_MCP(new_costs, mcp_alpha, offsets=offsets)
    mcp.find_costs([start], [end])
    route = mcp.traceback(end)

    return image_down, route    

def width_finding(image, ggm_sigma = 1.03907545, sig_per = 61.3435119, sig_growth_rate = 2.42541565, alpha = 1.00167702, mcp_alpha = 1.02251872):

    """From the image, find the widths
    NOTE: assume the image is the warped image
    with the width of the image = int(center_tck[0][-1]//5)
    This is the same width as the spline view of the pose annotator

    Parameters:
        image: image of a straightened worm to get the widths from
        width_tck: tcks that give the widths for the worm
        params: list of the parameter values in the order [ggm sigma, sigmoid percentile, 
                sigmoid growth rate, alpha for the penalties]
    """
    #normalize the image based on mode
    #print(ggm_sigma,sig_per,sig_growth_rate,alpha)
    #down sample the image
    #image_down = image
    #image_down = scale_image(image)
    #the centerline is half of the width
    image_down = pyramid.pyr_down(image, downscale=2)
    #image_down = image.astype(np.float32)

    #get the gradient
    gradient = ndimage.filters.gaussian_gradient_magnitude(image_down, ggm_sigma)
    y_grad = ndimage.filters.gaussian_gradient_magnitude(image_down, ggm_sigma)
    top_ten = np.percentile(gradient, sig_per )
    gradient = sigmoid(gradient, gradient.min(), top_ten, gradient.max(), sig_growth_rate)
    sig = gradient
    gradient = gradient.max()-abs(gradient)
    
    #get the widths from the width_tck and then get the distance matrix of the
    #widths to the centerline of the worm
    #NOTE: widths are the avg widths from the pca
    widths = interpolate.spline_interpolate(avg_width_tck, image_down.shape[0])
    widths = widths/2
    #widths = interpolate.spline_interpolate(width_tck, image_down.shape[0])
    #widths = widths/2 #need to divide by the downscale factor to get the right pixel values

    #penalizing the costs for being too far from the widths
    #half_distance_matrix = abs(np.subtract.outer(widths, np.arange(0, image_down.shape[1]/2)))
    
    
    #calculate the penalty
    #we want to penalize things that are farther from the average widths, so we square the distance
    distance_matrix = np.flip(abs(np.subtract.outer(widths, np.arange(0, image_down.shape[1]))), 1)
    penalty = alpha*(distance_matrix)
    new_costs = gradient+penalty
    #new_costs = gradient
    #set start and end points for the traceback
    start = (0, int((image_down.shape[1]-1)-widths[0]))
    end = (len(widths)-1, int((image_down.shape[1]-1)-widths[-1]))

    #start = (0, int((image_down.shape[1]/2)-widths[0]))
    #end = (len(widths)-1, int((image_down.shape[1]/2)-widths[-1]))
    #print(start, end)
    #print(new_costs.shape)
    offsets= [(1,-1),(1,0),(1,1)]
    #mcp = graph.MCP(new_costs, offsets=offsets, fully_connected=True)
    mcp = Smooth_MCP(new_costs, mcp_alpha, offsets=offsets)
    costs, _ = mcp.find_costs([start], [end])
    #print(costs.shape)
    route = mcp.traceback(end)
    #visualize things
    new_widths_image = make_traceback_image(route, image_down.shape)
    #width_image = visualize_widths(widths, image_down.shape)
    return (image_down, route, new_widths_image, new_costs, y_grad, sig)

def visualize_widths(widths, shape, top=True):
    #print(shape)
    image = np.zeros(shape)
    if top:
        #image[[np.arange(0, len(widths)), (widths+shape[1]/2).astype(np.uint8)]]=1
        image[[np.arange(0, len(widths)), widths.astype(np.uint8)]]=1
    else:
        image[[np.arange(0, len(widths)), widths.astype(np.uint8)]]=1
    #image[[np.arange(0,len(widths)),widths.astype(np.uint8)]]=1
    return image

def make_traceback_image(traceback, shape):
    #make it so we can visualize things
    image = np.zeros(shape)
    image[list(np.transpose(traceback))]=1
    return image

def circle_mask(cx, cy, r, shape):
    cx, cy, r = int(cx * shape[0]), int(cy * shape[1]), int(r * shape[0])
    path = celiagg.Path()
    path.ellipse(cx, cy, r, r)
    return worm_spline._celiagg_draw_mask(shape, path, antialias=False)

def tenX_mask(img_shape):
    """Since the 10x images have weird vignettes, generate a mask for
    the vignette
    """
    return circle_mask(*histogram_mask.HistogramMask.DEFAULT_MASKS[0.7], shape=img_shape)

def generate_width_spline(traceback, shape):
    '''From the traceback, generate a new
    width spline.
    '''
    #since the centerline is at x=0, we just need the
    #y values to get the widths
    _, widths = np.where(traceback)
    widths = shape[1]-widths-1
    x_vals = np.linspace(0,1, len(widths))
   
    smoothing = 0.2*len(widths)
    
    tck = interpolate.fit_nonparametric_spline(x_vals, widths, smoothing=smoothing)
    return tck

def scale_image(image, mag='5x'):
    if mag is '10x':
        mask = tenX_mask(image.shape).astype(bool)
        bf = image[mask]
        mode = np.bincount(bf.flat)[1:].argmax()+1
        bf = image.astype(np.float32)
        bf -= 200
        bf *= (24000-200) / (mode-200)
        bf8 = colorize.scale(bf, min=600, max=26000, gamma=1, output_max=255)

    else:
        mode = np.bincount(image.flat)[1:].argmax()+1
        bf = image.astype(np.float32)
        bf -= 200
        bf *= (24000-200) / (mode-200)
        bf8 = colorize.scale(bf, min=600, max=26000, gamma=0.72, output_max=255)
            
    return bf8

class Smooth_MCP(graph.MCP_Flexible):
    """Custom MCP class to weight different route possibilities.
    Penalize sharp changes to make the end widths a little smoother
    """
    def __init__(self, costs, alpha, offsets=None):
        graph.MCP_Flexible.__init__(self, costs, offsets=offsets)
        self.alpha = alpha

    def travel_cost(self, old_cost, new_cost, offset_length):
        """Override method to smooth out the traceback
        """
        penalty = 1 if offset_length == 1 else self.alpha
        return new_cost*penalty 




