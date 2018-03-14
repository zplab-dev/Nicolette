import numpy
import pathlib
import re
import datetime
from zplib import datafile
from ris_widget import ris_widget
import PyQt5.Qt as Qt
import PyQt5.QtGui as QtGui
import glob
import sys
import freeimage
import _pickle as pickle
import numpy as np
from scipy import ndimage
from zplib.image import mask
from zplib.image import active_contour
from zplib.curve import interpolate
from zplib.curve import spline_geometry
from zplib.curve import geometry
from zplib import pca
from ris_widget.overlay import free_spline



"""def load_worm(rw, mask_dir, keyword = '*mask.png'):
    print("loading worm: "+mask_dir)

    mask_dir = pathlib.Path(mask_dir)
    masks = mask_dir.glob(keyword)

    for i in sorted(masks):
        im=freeimage.read(i)
        im=im>0
        sx, sy=ndimage.find_objects(im)[0]
        x = slice(max(0, sx.start-50),min(im.shape[0], sx.stop+50))
        y = slice(max(0, sy.start-50),min(im.shape[1], sy.stop+50))
        img = im[x,y]
        bf = freeimage.read(straighten_worms.get_bf_image(i))
        crop_img = bf[x,y]
        m, skel, centerline, center_dist, med_axis, tb =skeleton.skel_and_centerline(img)
        #rw.flipbook_pages.append([img, m, skel, centerline, center_dist])
        rw.flipbook_pages.append([crop_img,img, m, skel, centerline, center_dist])
        rw.flipbook_pages[-1].spline_data = tb
        rw.flipbook_pages[-1].dist_data = center_dist
        rw.flipbook_pages[-1].img_path = [i, sx, sy]
        rw.flipbook_pages[-1].med_axis = med_axis
        rw.flipbook_pages[-1].name = str(i)"""

def load_splines(rw, spline_dict, mask_dir, keyword = '*mask.png'):
    '''Load the worms and their splines
    '''
    tck_dict = pickle.load(open(spline_dict, 'rb'))

    mask_dir = pathlib.Path(mask_dir)
    masks = mask_dir.glob(keyword)

    for i in sorted(masks):
        #grab splines
        tcks = tck_dict[str(i)]
        #tcks = tck_dict[timepoint]
        #load images
        rw.add_image_files_to_flipbook([i])
        rw.flipbook_pages[-1].spline_data = tcks
        rw.flipbook_pages[-1].img_path = i

def load_worm_from_spline(rw, spline_path, img_dir, keyword='*bf.png'):
    '''Load worms and their splines
    '''
    tck_dict = pickle.load(open(spline_path, 'rb'))
    img_dir = pathlib.Path(img_dir)

    #annotations to be used for the spline_view/spline_editor
    annotations = [{'tck':(ctck, wtck)} for _,ctck, wtck in sorted(tck_dict.values())]
    #annotations = [{'centerline':ctck, 'widths':wtck} for ctck, wtck in tck_dict.values()]
    #print(annotations)
    for timepoint in sorted(tck_dict.keys()):
        img_path = list(img_dir.glob(timepoint+keyword))
        #print(str(img_path))
        rw.add_image_files_to_flipbook(img_path)

    return annotations

def load_worm(rw, spline_dir, work_dir, worm_idx, keyword = '*bf.png'):
    '''Load a worm from spline to rw
    '''
    spline_dir = pathlib.Path(spline_dir)
    img_dir = pathlib.Path(work_dir)
    #print(work_dir)
    subdir = list(img_dir.glob('*'))[worm_idx]
    worm_id = subdir.name
    print("Loading worm " + worm_id)
    spline_path = list(spline_dir.glob(worm_id+'_tck.p'))[0]
    #load_splines(rw, spline_path, subdir, keyword = '*bf.png')
    annotations = load_worm_from_spline(rw, spline_path, subdir, keyword = keyword)
    if hasattr(rw,'annotator'):
        rw.annotator.all_annotations = annotations
    """for sub_dir in list(img_dir.glob('*'))[worm_idx]:
                    worm_id = sub_dir.name
                    spline_path = list(spline_dir.glob(worm_id+'_tck.p'))[0]
                    #print(str(spline_path))
                    load_splines(rw, spline_path, sub_dir, keyword = '*bf.png')"""

    return annotations

def load_all_splines(rw, spline_dict, img_dir, avg_width_tck, keyword='**/*bf.png'):
    tck_dict = pickle.load(open(spline_dict, 'rb'))

    for i, tck in sorted(tck_dict.items()):
        #timepoint = i.name.split(' ')[0]
        #grab splines
        #tcks = tck_dict[str(i)]
        #tcks = tck_dict[timepoint]
        #load images
        rw.add_image_files_to_flipbook([i])
        rw.flipbook_pages[-1].spline_data = (tck[0], avg_width_tck)
        rw.flipbook_pages[-1].img_path = i

def smooth_width_from_pca(width_tck, pca_stuff):
    '''Given a width_tck, use the pca to smooth out the edges
    '''
    #Go from PCA -> real widths and back (real widths -> PCA)
    #Widths -> PCA
    mean, pcs, norm_pcs, variances, total_variance,positions, norm_positions = pca_stuff
    widths = interpolate.spline_interpolate(width_tck, 100) 
    projection = pca.pca_decompose(widths, pcs, mean)[:3]
    #PCA -> Widths
    smoothed_widths = pca.pca_reconstruct(projection, pcs[:3], mean)
    #make new width spline
    smoothed_width_tck = interpolate.fit_nonparametric_spline(np.linspace(0,1, len(smoothed_widths)), smoothed_widths)
    return smoothed_width_tck

def generate_width_from_pca(projection, pca_stuff):
    '''If you are given a projection from pca (3 components),
    then generate a width_tck from it
    '''
    widths = pca.pca_reconstruct(projection, pcs[:3], mean)
    #make new width spline
    width_tck = interpolate.fit_nonparametric_spline(np.linspace(0,1, len(smoothed_widths)), smoothed_widths)
    return width_tck

def save_splines(rw, export_file):
    '''Save splines from risWidget as a pickle file
    '''
    spline_dict = {}
    for page in rw.flipbook_pages:
        timepoint = page.name.split(" ")[0]
        spline_dict[timepoint]=(page.annotations['centerline'], page.annotations['widths'])

    pickle.dump(spline_dict, open(export_file, 'wb'))

class SplineLoad:
    '''Class that can be used to load splines into ris_widget and then
    save them out to the same spline_path. This will be used when trying to edit
    splines and save them for later
    '''
    def __init__(self, rw, spline_dir, work_dir):
        '''spline_dir is the directory where all the tck dictionaries are saved
        work_dir is the working directory where all the masks/images should be
        NOTE: the code is built on the assumptions that they are in the format provided
        in Holly's experiments and assuming that you used the spline_generator code
        before trying to edit the splines
        '''
        self.rw = rw
        self.spline_dir = pathlib.Path(spline_dir)
        self.work_dir = pathlib.Path(work_dir)
        self.current_worm_idx = 0
        self.subdir = list(self.work_dir.glob('*'))
        self.actions = []
        
        self._add_action('prev', Qt.Qt.Key_BracketLeft, lambda: self.load_next_worm(-1))
        self._add_action('next', Qt.Qt.Key_BracketRight, lambda: self.load_next_worm(1))
        #self._add_action('Save Annotations', QtGui.QKeySequence('Ctrl+S'),self.save_annotations)
        self.load_worm(self.current_worm_idx)


    def _add_action(self, name, key, function):
        action = Qt.QAction(name, self.rw.qt_object)
        action.setShortcut(key)
        self.rw.qt_object.addAction(action)
        action.triggered.connect(function)
        self.actions.append(action)


    def load_next_worm(self, offset):
        '''Load a new worm!
        '''
        if len(self.rw.flipbook_pages)>0:
            #clear the flipbook
            self.rw.flipbook_pages.clear()

        #load in new images
        #if we are at the beginning of the list or end of the list don't load a new worm
        if self.current_worm_idx+offset>-1 and self.current_worm_idx+offset<len(self.subdir):
            self.load_worm(self.current_worm_idx+offset)
            self.current_worm_idx += offset


    def load_worm_from_spline(self, spline_path, img_dir, keyword='*bf.png'):
        '''Load worms and their splines
        '''
        
        tck_dict = pickle.load(open(spline_path, 'rb'))
        img_dir = pathlib.Path(img_dir)

        #annotations to be used for the spline_view/spline_editor
        annotations = [{'tck':tck} for tp, tck in sorted(tck_dict.items())]
        #annotations = [{'centerline':ctck, 'widths':wtck} for ctck, wtck in tck_dict.values()]
        #print(annotations)
        for timepoint in sorted(tck_dict.keys()):
            img_path = list(img_dir.glob(timepoint+keyword))
            #print(str(img_path))
            self.rw.add_image_files_to_flipbook(img_path)

        return annotations

    def load_worm(self, worm_idx, keyword = '*bf.png'):
        '''Load a worm from spline to rw
        '''
        subdir = self.subdir[worm_idx]
        worm_id = subdir.name
        print("Loading worm " + worm_id)
        spline_path = list(self.spline_dir.glob(worm_id+'_tck.p'))[0]
        #load_splines(rw, spline_path, subdir, keyword = '*bf.png')
        annotations = self.load_worm_from_spline(spline_path, subdir, keyword = keyword)
        if hasattr(self.rw,'annotator'):
            self.rw.annotator.all_annotations = annotations


class SplineAnnotate:
    '''Generate an annotator to determine which splines are best
    '''

    def __init__(self, rw):
        self.rw = rw
        ok = ris_widget.dock_widgets.annotator.BoolField('Spline OK', default=True)
        fields = [ok]
        self.annotator = ris_widget.dock_widgets.Annotator(self.rw, fields)
    
    def get_good_splines(self):
        #get annotations
        annotate_list = self.annotator.all_annotations
        #

class SplineView:
    '''Class that allows splines to be drawn on RisWidget
    '''

    def __init__(self, rw, pca_stuff, avg_width_tck, save_dir):
        
        self.rw = rw
        self.flipbook = rw.flipbook
        self.flipbook.current_page_changed.connect(self.page_changed)
        self.display_path = None
        self.display_path1 = None
        self.pca_stuff = pca_stuff
        #self.spline_dict = spline_dict
        self.avg_width_tck = avg_width_tck
        self.undo = None

        self.layout = Qt.QFormLayout()
        spline_widget = Qt.QWidget()
        spline_widget.setLayout(self.layout)

        self.sv = Qt.QPushButton("View Spline")
        self.pca_button = Qt.QPushButton("Use PCA Average")
        self.undo_button = Qt.QPushButton("Undo")
        self.save_splines = Qt.QPushButton("Save Splines")
        self.save_dir = save_dir
        self.layout.addRow(self.sv)
        self.layout.addRow(self.pca_button)
        self.layout.addRow(self.undo_button)
        self.rw.flipbook.layout().addWidget(spline_widget)

        self.sv.clicked.connect(self._on_sv_clicked)
        self.pca_button.clicked.connect(self._on_pca_clicked)
        self.undo_button.clicked.connect(self._on_undo_clicked)

    def _on_undo_clicked(self):
        temp = self.rw.flipbook.current_page.annotations['widths']
        self.rw.flipbook.current_page.annotations['widths'] = self.undo
        self.undo = temp
        
        self.update_view()

    def _on_save_clicked(self):

        return

    def _on_pca_clicked(self):
        #change the widths_tck to use the average pca values
        new_width_tck = smooth_width_from_pca(self.avg_width_tck, self.pca_stuff)
        self.undo = self.rw.flipbook.current_page.annotations['widths']
        self.rw.flipbook.current_page.annotations['widths'] = new_width_tck
        self.update_view()


    def _on_sv_clicked(self):
        #if the spline isn't in view then view it, otherwise delete it
        if self.display_path is not None:
            self.remove_spline(self.display_path)
            self.display_path = None
        if self.display_path1 is not None:
            self.remove_spline(self.display_path1)
            self.display_path1 = None

        else:
            self.view_spline(self.flipbook)


    def update_view(self):
        '''Sometimes we want to update a view after pressing a button
        '''
        if self.display_path is not None:
            self.remove_spline(self.display_path)
        if self.display_path1 is not None:
            self.remove_spline(self.display_path1)

        self.view_spline(self.flipbook)

    def disconnect(self):
        self.flipbook.page_changed.disconnect(self.page_changed)

    def plot_spline(self, tck, rgba):
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
        display_path = Qt.QGraphicsPathItem(path, parent=self.rw.image_scene.layer_stack_item)
        pen = Qt.QPen(Qt.QColor(*rgba, 100))
        pen.setWidth(3)
        pen.setCosmetic(True)
        display_path.setPen(pen)
        return display_path

    def plot_polygon(self, tck, width_tck, rgba):
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
        display_path = Qt.QGraphicsPathItem(path, parent=self.rw.image_scene.layer_stack_item)
        pen = Qt.QBrush(Qt.QColor(*rgba,90))
        #pen.setColor(Qt.QColor(*rgba))
        #pen.setWidth(1)
        #pen.setCosmetic(True)
        display_path.setBrush(pen)
        return display_path

    def view_spline(self, flipbook):
        current_idx = self.flipbook.current_page_idx

        #center_tck, width_tck = self.rw.flipbook_pages[current_idx].spline_data
        '''
        img = self.rw.flipbook_pages[current_idx].name
        center_tck = self.spline_dict['/mnt/lugia_array/data/human_masks/'+str(img)][0]
        old_width_tck = self.spline_dict['/mnt/lugia_array/data/human_masks/'+str(img)][1]
        '''
        center_tck = self.rw.flipbook.current_page.annotations['centerline']
        old_width_tck = self.rw.flipbook.current_page.annotations['widths']

        width_tck=smooth_width_from_pca(old_width_tck, self.pca_stuff)
        #traceback = self.flipbook.pages[current_idx].spline_data
        #dist = self.flipbook.pages[current_idx].dist_data
        #print("worm length: ",len(traceback))
        #tck = center_spline(traceback, dist)
        #print("tck length:",len(tck))
        #width_tck = width_spline(traceback, dist)

        #new_tck, new_width_tck = extrapolate_head(tck, width_tck)

        #display path for the centerline
        #self.display_path = self.plot_spline(center_tck, (107, 0, 183))
        #display path for the outline
        self.display_path1 = self.plot_polygon(center_tck, width_tck, (43, 141, 247))

    def remove_spline(self, display_path):
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
        self.rw.image_scene.removeItem(display_path)


    def page_changed(self, flipbook):
        '''Displays spline/ploygon on the current flipbook page
        '''
        #print(self.flipbook.current_page_idx)

        #current_idx = self.flipbook.current_page_idx
        if self.display_path is not None:
            self.remove_spline(self.display_path)
            #display_path = None
        if self.display_path1 is not None:
            self.remove_spline(self.display_path1)
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


class SplineEdit(free_spline.FreeSpline):
    def __init__(self, ris_widget, pca, color=Qt.Qt.green, geometry=None, on_geometry_change=None):
        
        self._width_tck = None
        self._display_path = None
        self.layout = Qt.QFormLayout()
        self.rw = ris_widget
        self.pca_stuff = pca
        self.undo = None
        self.avg_width_tck = self._generate_pca_avg()
        spline_widget = Qt.QWidget()
        spline_widget.setLayout(self.layout)

        self.sv = Qt.QPushButton("View Spline")
        self.pca_button = Qt.QPushButton("Use PCA Average")
        self.undo_button = Qt.QPushButton("Undo")
        self.layout.addRow(self.sv)
        self.layout.addRow(self.pca_button)
        self.layout.addRow(self.undo_button)

        self.rw.flipbook.layout().addWidget(spline_widget)
        self.sv.clicked.connect(self._on_sv_clicked)
        self.pca_button.clicked.connect(self._on_pca_clicked)
        self.undo_button.clicked.connect(self._on_undo_clicked)
        super().__init__(ris_widget, color, geometry, on_geometry_change)

    
    def _on_sv_clicked(self):
        #if the spline isn't in view then view it, otherwise delete it
        if self._display_path is not None:
            self._remove_spline()
            self._display_path = None
        else:
            #print("display_path is None")
            self._view_spline()
            #print("new display_path", self._display_path)

   
    def _on_undo_clicked(self):
        temp = self._width_tck
        self._width_tck = self.undo
        self.undo = temp
        self._update_view()


    def _on_pca_clicked(self):
        #change the widths_tck to use the average pca values
        new_width_tck = smooth_width_from_pca(self.avg_width_tck, self.pca_stuff)
        self.undo = self._width_tck
        self._width_tck = new_width_tck
        self._update_view()
   
    def _generate_pca_avg(self):
        mean, pcs, norm_pcs, variances, total_variance,positions, norm_positions = self.pca_stuff
        avg_width_positions = pca.pca_reconstruct([0,0,0], pcs[:3], mean)
        avg_width_tck = interpolate.fit_nonparametric_spline(np.linspace(0,1, len(avg_width_positions)), avg_width_positions)
        return avg_width_tck

    def remove(self):
        super().remove()


    @property
    def geometry(self):
        return (self._tck, self._width_tck)


    @geometry.setter
    def geometry(self, tcks):
        #if self._display_path is not None:
        #  self._remove_spline()

        if tcks is not None:
            #print(tcks)
            center_tck, width_tck = tcks    
            self._width_tck = width_tck
        else:
            center_tck = None
        free_spline.FreeSpline.geometry.fset(self, center_tck)

        #if self._tck is not None and self._width_tck is not None:
        self._update_view()
        
    def _update_view(self):
        if self._display_path is not None:
            self._remove_spline()
        if self._tck is not None and self._width_tck is not None:
            self._view_spline()

        self._geometry_changed()

    def _reverse_spline(self):
        ct, cc, ck = self._tck
        wt, wc, wk = self._width_tck

        new_center_t=np.absolute(ct[-1]-ct[::-1])
        new_center_c=cc[::-1] 
        new_width_t=(wt[-1]-wt[::-1])                              
        new_width_c=wc[::-1]

        new_center_tck = (np.array(new_center_t), np.array(new_center_c), ck)
        new_width_tck = (np.array(new_width_t), np.array(new_width_c), wk)
        self.set_tck(new_center_tck)
        self.width_tck = new_width_tck
        self._update_view()


    def _plot_polygon(self, tck, width_tck, rgba):
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
        display_path = Qt.QGraphicsPathItem(path, parent=self.rw.image_scene.layer_stack_item)
        pen = Qt.QBrush(Qt.QColor(*rgba,90))
        #pen.setColor(Qt.QColor(*rgba))
        #pen.setWidth(1)
        #pen.setCosmetic(True)
        display_path.setBrush(pen)
        return display_path

    def _smooth_width_from_pca(self, width_tck):
        '''Given a width_tck, use the pca to smooth out the edges
        '''
        #Go from PCA -> real widths and back (real widths -> PCA)
        #Widths -> PCA
        mean, pcs, norm_pcs, variances, total_variance,positions, norm_positions = self.pca_stuff
        widths = interpolate.spline_interpolate(width_tck, 100) 
        projection = pca.pca_decompose(widths, pcs, mean)[:3]
        #PCA -> Widths
        smoothed_widths = pca.pca_reconstruct(projection, pcs[:3], mean)
        #make new width spline
        smoothed_width_tck = interpolate.fit_nonparametric_spline(np.linspace(0,1, len(smoothed_widths)), smoothed_widths)
        return smoothed_width_tck

    def _view_spline(self):   
        center_tck = self._tck
        old_width_tck = self._width_tck
        #width_tck = self._width_tck
        width_tck=self._smooth_width_from_pca(old_width_tck)
        #TODO: add in the pca smoothing for funsies
        #width_tck=smooth_width_from_pca(old_width_tck, self.pca_stuff)
        #traceback = self.flipbook.pages[current_idx].spline_data
        #dist = self.flipbook.pages[current_idx].dist_data
        #print("worm length: ",len(traceback))
        #tck = center_spline(traceback, dist)
        #print("tck length:",len(tck))
        #width_tck = width_spline(traceback, dist)

        #new_tck, new_width_tck = extrapolate_head(tck, width_tck)

        #display path for the centerline
        #self.display_path = self.plot_spline(center_tck, (107, 0, 183))
        #display path for the outline
        self._display_path = self._plot_polygon(center_tck, width_tck, (43, 141, 247))

    def _remove_spline(self):
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
        self.rw.image_scene.removeItem(self._display_path)





