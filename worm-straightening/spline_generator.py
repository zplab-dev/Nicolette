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
import _pickle as pickle
import straighten_worms

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
        

    print(index)
    print("length: "+str(len(traceback)))
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
    points = np.array(list(np.transpose(traceback))).T
    #print(points.shape)
    widths = distances[list(np.transpose(traceback))]

    if smoothing is None:
        smoothing = 0.2*len(widths)

    #create splines for the first and last few points
    begin_tck = interpolate.fit_spline(points[:10], smoothing=smoothing, order = 1)
    begin_xys = interpolate.spline_evaluate(begin_tck, np.linspace(-widths[0], 0, int(widths[0]), endpoint=False))
    #print(begin_xys.shape)
    end_tck = interpolate.fit_spline(points[-10:], smoothing =smoothing, order = 1)
    tmax = end_tck[0][-1]
    end_xys = interpolate.spline_evaluate(end_tck, np.linspace(tmax+tmax/widths[-1], tmax+widths[-1], int(widths[-1])))
    #print(end_xys.shape)
    new_points = np.concatenate((begin_xys, points, end_xys))
    #print(new_points.shape)
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

    for img_path in list(mask_dir.glob("*mask.png")):
        #load in images
        img = freeimage.read(img_path)
        img=img>0

        timepoint = img_path.stem.split(' ')[0]

        #generate splines
        traceback, medial_axis = skel_and_centerline(img)
        center_tck = center_spline(traceback, medial_axis)
        width_tck = width_spline(traceback, medial_axis)

        #add to spline_dict
        spline_dict[timepoint] = (center_tck, width_tck)

    #save spline_dict out
    output_name = export_dir+"/"+mask_dir.name + "_tck.p"
    print("Pickling spline_dict and saving to: " + str(output_name))

    pickle.dump(spline_dict, open(output_name, 'wb'))

def warp_worms(img_dir, spline_pickle, export_dir, keyword="*gfp.png"):
    '''Once a spline dict has been created and exported in a pickle
    form, use it to warp worms and save the warped images into the export_dir

    NOTE: Assuming that the img_dir is where the spline_dict images came from
    '''

    spline_dict = pickle.load(open(spline_pickle, 'rb'))
    img_dir = pathlib.Path(img_dir)

    for timepoint in spline_dict.keys():
        img_path = list(img_dir.glob(timepoint+keyword))[0]
        #timepoint = img_path.stem.split(' ')[0]
        center_tck, width_tck = spline_dict[timepoint]

        #assuming we want a bf_image to check our straightening splines
        #NOTE: assuming the bf and gfp images are in the same directory
        bf_image = straighten_worms.get_bf_image(img_path)
        output_name = export_dir+"/"+img_path.stem+" warped.png"
        bf_warp_name = export_dir+"/"+bf_image.stem+" warped.png"

        #save both gfp and bf warps
        straighten_worms.warp_image(center_tck, width_tck, freeimage.read(img_path), output_name)
        straighten_worms.warp_image(center_tck, width_tck, freeimage.read(bf_image), bf_warp_name)


