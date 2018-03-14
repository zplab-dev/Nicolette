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
    print("length: "+str(len(traceback)))
    #center=[]
    center=generate_centerline(traceback, skeleton.shape)
    #tck = generate_spline(traceback, 15)
    return center,traceback,endpoints


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


def generate_centerline_from_points(points, skeleton):
    '''Generate a traceback of the centerline from a list of points
    along the skeleton that you want the centerline to go through
    
    Parameters:
    ------------
    points: list of 2-d tuples
        List of indices that the centerline will go through. 
        Indices must be given in the order that the centerline should 
        encounter each point
    skeleton: array_like (cast to booleans) shape (n,m) 
        Binary image of the skeleton of the worm mask

    Returns:
    -----------
    traceback: list of 2-d tuples
        List of indices associated with the centerline, starting with
        one of the endpoints of the centerline and ending with the ending
        index of the centerline. 
    '''
    skel_path = np.transpose(np.where(skeleton))
    skeleton = skeleton.astype(float)
    skeleton[skeleton==0]=np.inf
    #create mcp object
    mcp = skg.MCP(skeleton)
    traceback=[]
    for i in range(0, len(points)-1):
        start = points[i]
        end = points[i+1]
        print("start: ",start," end: ", end)
        #print(skeleton[start])
        #print(skeleton[end])
        if np.all(np.isinf(skeleton[points[i]])):
            #print("start not in skeleton")
            start = geometry.closest_point(start, skel_path)[1]
        if np.all(np.isinf(skeleton[end])):
            #print("end: ", end)
            end = geometry.closest_point(end, skel_path)[1]

        
        costs = mcp.find_costs([start], [end])
        tb = mcp.traceback(end)
        traceback.extend(tb[:-1])

    return traceback


def generate_splines_from_points(points, mask):
    '''Generate centerline spline and width splines along
    a route of points you want the splines/centerlines to go through
    
    Parameters:
    ------------
    points: list of 2-d tuples
        List of indices that the centerline will go through. 
        Indices must be given in the order that the centerline should 
        encounter each point
    mask: array_like (cast to booleans) shape (n,m) 
        Binary image of the worm mask

    Returns:
    -----------
    center_tck: parametric spline tuple
        spline tuple for spline corresponding to the centerline 
    width_tck: nonparametric spline tuple
        spline tuple corresponding to the widths along the
        centerline of the worm
    '''

    traceback, distances = generate_center_med_axis_from_points(points, mask)
    center_tck = center_spline(traceback, distances)
    width_tck = width_spline(traceback, distances)

    return center_tck, width_tck

def generate_center_med_axis_from_points(points, mask):
    '''Generate medial axis transform for a centerline
        from a list of points that the centerline will go through.
        This is used to generate all the widths along a particular centerline
    
    Parameters:
    ------------
    points: list of 2-d tuples
        List of indices that the centerline will go through. 
        Indices must be given in the order that the centerline should 
        encounter each point
    mask: array_like (cast to booleans) shape (n,m) 
        Binary image of the worm mask

    Returns:
    -----------
    traceback: list of 2-d tuples
        List of indices associated with the centerline, starting with
        one of the endpoints of the centerline and ending with the ending
        index of the centerline. 
    distances: ndarray shape(n,m)
        Distance transform from the medial axis transform of the worm mask
    '''
    skeleton, med_axis = medial_axis(mask, return_distance=True)
    traceback = generate_centerline_from_points(points, skeleton)
    centerline = generate_centerline(traceback, mask.shape)
    distances = centerline*med_axis

    return traceback, distances


def update_splines(traceback, mask):
    '''Generate center_tck and width_tck from a new traceback

    Parameters:
    ------------
    traceback: list of 2-d tuples
        List of indices associated with the centerline, starting with
        one of the endpoints of the centerline and ending with the ending
        index of the centerline. 
    mask: array_like (cast to booleans) shape (n,m) 
        Binary image of the worm mask

    Returns:
    -----------
    center_tck: parametric spline tuple
        spline tuple for spline corresponding to the centerline 
    width_tck: nonparametric spline tuple
        spline tuple corresponding to the widths along the
        centerline of the worm
    '''
    skeleton, med_axis = medial_axis(mask, return_distance=True)
    centerline = generate_centerline(traceback, mask.shape)
    distances = centerline*med_axis
    center_tck = center_spline(traceback, distances)
    width_tck = width_spline(traceback, distances)

    return center_tck, width_tck


def extrapolate_head(tck, width_tck, smoothing=None):
    '''Since the width/length splines don't get to the end of the worm,
    try to extrapolate the widths and such to the end of the worm mask

    NOTE: Not really used in the spline-fitting pipeline
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
    return mask, skeleton, centerline, center_dist, med_axis, traceback

def plot_spline(rw, tck, rgba):
    '''Plot the spline on RisWidget

    Parameters:
    ------------
    rw: RisWidget Object
        Reference to the RisWidget object you want to display
        the spline on
    tck: parametric spline tuple
        Spline tuple to plot
    rgba: 3-d tuple of ints
        RGBA values to color the spline

    Returns:
    -----------
    display_path: QGraphicsPathItem
        Reference to the display item displaying the spline
    '''
    bezier_elements = interpolate.spline_to_bezier(tck)
    path = Qt.QPainterPath()
    path.moveTo(*bezier_elements[0][0])
    for (sx, sy), (c1x, c1y), (c2x, c2y), (ex, ey) in bezier_elements:
        path.cubicTo(c1x, c1y, c2x, c2y, ex, ey)
    display_path = Qt.QGraphicsPathItem(path, parent=rw.image_scene.layer_stack_item)
    pen = Qt.QPen(Qt.QColor(*rgba, 100))
    pen.setWidth(3)
    pen.setCosmetic(True)
    display_path.setPen(pen)
    return display_path

def plot_polygon(rw, tck, width_tck, rgba):
    '''Plot the full polygon on RisWidget
    
    Parameters:
    ------------
    rw: RisWidget Object
        Reference to the RisWidget object you want to display
        the spline on
    tck: parametric spline tuple
        Spline tuple of the centerline to plot
    width_tck: nonparametric spline tuple
        Spline tuple of the widths along the centerline to plot
    rgba: 3-d tuple of ints
        RGBA values to color the spline

    Returns:
    -----------
    display_path: QGraphicsPathItem
        Reference to the display item displaying the spline
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

def update_spline_from_points_in_rw(rw, points):
    '''update the spline data and such for the current flipbook page
    Nice way to update things when looking at the splines

    Parameters:
    ------------
    rw: RisWidget Object
        Reference to the RisWidget object you want to display
        the spline on
    points: list of 2-d tuples
        List of indices that the centerline will go through. 
        Indices must be given in the order that the centerline should 
        encounter each point 

    Returns:
    -----------
    '''
    current_idx = rw.flipbook.current_page_idx
    mask = rw.flipbook.pages[current_idx][1].data

    traceback, distances = generate_center_med_axis_from_points(points, mask)
    #update stuff in risWidget
    rw.flipbook.pages[current_idx].spline_data = traceback
    rw.flipbook.pages[current_idx].dist_data = distances

def remove_spline(rw, display_path):
    '''Remove the spline displayed using the display_path on RisWidget

    Parameters:
    ------------
    rw: RisWidget Object
        Reference to the RisWidget object you want to display
        the spline on
    display_path: QGraphicsPathItem
        Reference to the display item displaying the spline

    Returns:
    -----------
    '''
    rw.image_scene.removeItem(display_path)


class Spline_View:
    '''Class that allows splines to be drawn on RisWidget
    '''

    def __init__(self, rw):
        
        self.rw = rw
        self.flipbook = rw.flipbook
        self.flipbook.current_page_changed.connect(self.page_changed)
        self.display_path = None
        self.display_path1 = None

        self.layout = Qt.QFormLayout()
        spline_widget = Qt.QWidget()
        spline_widget.setLayout(self.layout)

        self.sv = Qt.QPushButton("View Spline")
        self.layout.addRow(self.sv)
        self.rw.flipbook.layout().addWidget(spline_widget)

        self.sv.clicked.connect(self._on_sv_clicked)

    def _on_sv_clicked(self):
        #if the spline isn't in view then view it, otherwise delete it
        if self.display_path is not None:
            remove_spline(self.rw, self.display_path)
            self.display_path = None
        if self.display_path1 is not None:
            remove_spline(self.rw, self.display_path1)
            self.display_path1 = None

        else:
            self.view_spline(self.flipbook)

    def disconnect(self):
        self.flipbook.page_changed.disconnect(self.page_changed)

    def view_spline(self, flipbook):
        current_idx = self.flipbook.current_page_idx

        traceback = self.flipbook.pages[current_idx].spline_data
        dist = self.flipbook.pages[current_idx].dist_data
        #print("worm length: ",len(traceback))
        tck = center_spline(traceback, dist)
        #print("tck length:",len(tck))
        width_tck = width_spline(traceback, dist)

        #new_tck, new_width_tck = extrapolate_head(tck, width_tck)

        #display path for the centerline
        self.display_path = plot_spline(self.rw, tck, (107, 0, 183))
        #display path for the outline
        self.display_path1 = plot_polygon(self.rw, tck, width_tck, (43, 141, 247))


    def page_changed(self, flipbook):
        '''Displays spline/ploygon on the current flipbook page
        '''
        #print(self.flipbook.current_page_idx)

        #current_idx = self.flipbook.current_page_idx
        if self.display_path is not None:
            remove_spline(self.rw, self.display_path)
            #display_path = None
        if self.display_path1 is not None:
            remove_spline(self.rw, self.display_path1)
            #display_path1 = None

        self.view_spline(self.flipbook)
        
        """traceback = self.flipbook.pages[current_idx].spline_data
                                dist = self.flipbook.pages[current_idx].dist_data
                                print("worm length: ",len(traceback))
                                tck = center_spline(traceback, dist)
                                #print("tck length:",len(tck))
                                width_tck = width_spline(traceback, dist)
                        
                                #new_tck, new_width_tck = extrapolate_head(tck, width_tck)
                        
                                #display path for the centerline
                                self.display_path = plot_spline(self.rw, tck, (107, 0, 183))
                                #display path for the outline
                                self.display_path1 = plot_polygon(self.rw, tck, width_tck, (43, 141, 247))"""





