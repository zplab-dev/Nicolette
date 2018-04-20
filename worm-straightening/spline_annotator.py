import pathlib
from ris_widget import ris_widget
import PyQt5.Qt as Qt
import PyQt5.QtGui as QtGui
import freeimage
import _pickle as pickle
from ris_widget import split_view
from ris_widget.qwidgets import annotator
import spline_utils
from elegant.gui import pose_annotation

class SplineAnnotator:
    '''This is a wrapper class with a few other functions to
    load up splines/images into RisWidget for editing and saving
    '''

    def __init__(self, rw, spline_dir, work_dir, pca, name='pose'):
        '''Initialize the SplineAnnotator object. This object will load
        in bf images of a worm with the previously generated splines.
        You can save the splines to use later using ctrl+s, you can load images of
        worms with left and right brackets
            
        Parameters:
        ------------
            rw: a ris_widget.RisWidget() instance
            spline_dir: the path to where all the pickle files containing the previously
                generated splines are
            work_dir: the path to bf images
                NOTE: the program assumes the names of the pickle files are the same as the 
                files in the work_dir. (ex. if the work_dir has a folder named '20170919_lin-04_GFP_spe-9 020',
                then the program expects the corresponding pickle file to be named '20170919_lin-04_GFP_spe-9 020_tck.p')
            pca: tuple containing the outputs from zplib.pca function
                tuple contains:
                (mean, pcs, norm_pcs, variances, total_variance,positions, norm_positions)
            name: name to give the PoseAnnotation object to look for in the annotations
                This is used to ensure the spline editor and the annotations play nicely

        Returns:
        -----------
        SplineAnnotator object instance
        '''
        self.rw = rw
        ws = pose_annotation.PoseAnnotation(rw, mean_widths=pca[0], width_pca_basis=pca[1][:4], name=name)
        self.rw.add_annotator([ws])
        self.spline_load = SplineLoad(rw, spline_dir, work_dir, name=name)
        self.actions = []
        
        self._add_action('save', QtGui.QKeySequence('Ctrl+S'),self.save_annotations)

    def _add_action(self, name, key, function):
        action = Qt.QAction(name, self.rw.qt_object)
        action.setShortcut(key)
        self.rw.qt_object.addAction(action)
        action.triggered.connect(function)
        self.actions.append(action)

    def save_annotations(self):
        subdir = self.spline_load.subdir[self.spline_load.current_worm_idx]
        worm_id = subdir.name
        save_file = list(self.spline_load.spline_dir.glob(worm_id+'_tck.p'))[0]
        print("Saving splines to:",save_file)
        self.save_splines(save_file)

    def save_splines(self, export_file):
        '''Save splines from risWidget as a pickle file
        '''
        spline_dict = {}
        for page in self.rw.flipbook_pages:
            timepoint = page.name.split(" ")[0]
            spline_dict[timepoint]=page.annotations['pose']

        with open(export_file, 'wb') as save_file: pickle.dump(spline_dict, save_file)
        save_file.close()
   



   

class SplineLoad:
    '''Class that can be used to load splines into ris_widget and then
    save them out to the same spline_path. This will be used when trying to edit
    splines and save them for later
    '''
    def __init__(self, rw, spline_dir, work_dir, name='pose'):
        '''Initialize the SplineLoad object. This object will load
        in bf images of a worm with the previously generated splines.
        You can load images of worms with left and right brackets.
            
        Parameters:
        ------------
            rw: a ris_widget.RisWidget() instance
            spline_dir: the path to where all the pickle files containing the previously
                generated splines are
            work_dir: the path to bf images
                NOTE: the program assumes the names of the pickle files are the same as the 
                files in the work_dir. (ex. if the work_dir has a folder named '20170919_lin-04_GFP_spe-9 020',
                then the program expects the corresponding pickle file to be named '20170919_lin-04_GFP_spe-9 020_tck.p')
            name: name to give the PoseAnnotation object to look for in the annotations
                This is used to ensure the spline editor and the annotations play nicely

        Returns:
        -----------
        SplineLoad object instance
        '''

        self.rw = rw
        self.spline_dir = pathlib.Path(spline_dir)
        self.work_dir = pathlib.Path(work_dir)
        self.current_worm_idx = 0
        self.subdir = list(self.work_dir.glob('*'))
        self.actions = []
        self.name = name
        
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
        '''Load a new worm! Using a specified offset.
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
        #print(img_dir)

        #annotations to be used for the spline_view/spline_editor
        annotations = [{self.name:tck} for tp, tck in sorted(tck_dict.items())]
        #annotations = [{'centerline':ctck, 'widths':wtck} for ctck, wtck in tck_dict.values()]
        #print(annotations)
        self.rw.flipbook.add_image_files(str(img_dir.joinpath(keyword)))

        #for timepoint in sorted(tck_dict.keys()):
        #    img_path = list(img_dir.glob(timepoint+keyword))
            #print(str(img_path))
        #    self.rw.add_image_files_to_flipbook(img_path)

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



