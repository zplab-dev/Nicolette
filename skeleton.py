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



def find_enpoints(skeleton):
    '''Use hit-and-miss tranforms to find the endpoints
    of the skeleton
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
    return np.logical_or.reduce([ep1, ep2, ep3, ep4, ep5, ep6, ep7, ep8])

def find_centerline(skeleton):
    '''Find the centerline of the worm from the skeleton
    Outputs both the centerline values (traceback and image to view)
    and the spline interpretation of the traceback
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

def generate_centerline(traceback, shape):
    '''Generate a picture with the centerline defined
    '''
    img = np.zeros(shape)
    #x, y = zip(*traceback)
    img[list(np.transpose(traceback))]=1
    return img

def clean_mask(img):
    '''Clean spurious edges/unwanted things from the masks 
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

def generate_splines(traceback, distances, smoothing=None):
    '''generate the width and centerline splines
    NOTE: we will extrapolate the first/last few pixels to get the full length of the worm,
    since medial axis transforms/skeltons don't always go to the edge of the mask
    '''
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
    '''Calculate a spline for the widths
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

def extrapolate_head(tck, width_tck, smoothing=None):
    '''Since the width/length splines don't get to the end of the worm,
    try to extrapolate the widths and such to the end of the worm mask
    '''
    #get first and last points from the width_tck
    width_ends = interpolate.spline_interpolate(width_tck, 2)
    print(width_ends)
    #calculate new x's and y's that go to the end of the mask
    tmax = tck[0][-1]
    print("tmax: "+str(tmax))
    xys = interpolate.spline_evaluate(tck, np.linspace(-width_ends[0], tmax+width_ends[1], 600))
    #print(xys.shape)

    #interpolate the widths so that we can add on the end widths
    #NOTE: end widths will be zero since they are at the end of the mask
    widths = interpolate.spline_interpolate(width_tck, 600)
    new_widths = np.concatenate([[0], widths, [0]])
    #need to generate new x's to use to re-make the width splines with new_widths
    #new endpoint xs need to be reflective of where they are on the centerline
    new_xs = np.concatenate([[-widths[0]/tmax], np.linspace(0,1,600), [1+widths[-1]/tmax]])

    #re-make the splines
    if smoothing is None:
        smoothing = 0.2*len(new_widths)
    
    new_tck = interpolate.fit_spline(xys, smoothing=smoothing)

    new_width_tck = interpolate.fit_nonparametric_spline(new_xs, new_widths, smoothing=smoothing)
    return new_tck, new_width_tck

def skel_and_centerline(img):
    '''generate the skeleton and find the centerline
    img = array
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
    skeleton, distance = medial_axis(mask, return_distance=True)
    #med_axis = distance*skeleton
    #skeleton = skeletonize(mask)
    #find centerline
    centerline, traceback, endpoints = find_centerline(skeleton)
    distance = centerline*distance
    return mask, skeleton, centerline, distance, traceback, endpoints

def plot_spline(rw, tck, rgba):
    '''plot the spline on RisWidget
    '''
    bezier_elements = interpolate.spline_to_bezier(tck)
    path = Qt.QPainterPath()
    path.moveTo(*bezier_elements[0][0])
    for (sx, sy), (c1x, c1y), (c2x, c2y), (ex, ey) in bezier_elements:
        path.cubicTo(c1x, c1y, c2x, c2y, ex, ey)
    display_path = Qt.QGraphicsPathItem(path, parent=rw.image_scene.layer_stack_item)
    pen = Qt.QPen(Qt.QColor(*rgba, 100))
    pen.setWidth(2)
    pen.setCosmetic(True)
    display_path.setPen(pen)
    return display_path

def plot_polygon(rw, tck, width_tck, rgba):
    '''plot the full polygon on RisWidget
    '''
    left, right, outline = spline_geometry.outline(tck, width_tck)
    #print(left)
    path = Qt.QPainterPath()
    path.moveTo(*outline[0])
    for x,y in outline: 
           path.lineTo(x,y)
    path.closeSubpath()
    display_path = Qt.QGraphicsPathItem(path, parent=rw.image_scene.layer_stack_item)
    pen = Qt.QBrush(Qt.QColor(*rgba,90))
    #pen.setColor(Qt.QColor(*rgba))
    #pen.setWidth(1)
    #pen.setCosmetic(True)
    display_path.setBrush(pen)
    return display_path

def remove_spline(rw, display_path):
    '''remove the spline currently displayed on RisWidget
    '''
    rw.image_scene.removeItem(display_path)

class Spline_View:
    '''Class uses for looking at many splines at once
    '''

    def __init__(self, rw):
        self.rw = rw
        self.flipbook = rw.flipbook
        self.flipbook.current_page_changed.connect(self.page_changed)
        self.display_path = None
        self.display_path1 = None

    def disconnect(self):
        self.flipbook.page_changed.disconnect(self.page_changed)

    def page_changed(self, flipbook):
        #print(self.flipbook.current_page_idx)
        current_idx = self.flipbook.current_page_idx
        if self.display_path is not None:
            remove_spline(self.rw, self.display_path)
        if self.display_path1 is not None:
            remove_spline(self.rw, self.display_path1)
        
        traceback = self.flipbook.pages[current_idx].spline_data
        dist = self.flipbook.pages[current_idx].dist_data
        print("worm length: ",len(traceback))
        tck = generate_splines(traceback, dist)
        width_tck = width_spline(traceback, dist)

        #new_tck, new_width_tck = extrapolate_head(tck, width_tck)

        #display path for the centerline
        self.display_path = plot_spline(self.rw, tck, (165, 7, 144))
        #display path for the outline
        self.display_path1 = plot_polygon(self.rw, tck, width_tck, (43, 141, 247))





