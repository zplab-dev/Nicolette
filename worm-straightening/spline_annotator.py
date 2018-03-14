import pathlib
from ris_widget import ris_widget
import PyQt5.Qt as Qt
import PyQt5.QtGui as QtGui
import freeimage
import _pickle as pickle
from ris_widget import split_view
from ris_widget.qwidgets import annotator
import spline_utils

class SplineAnnotator:
    '''This is a wrapper class with a few other functions to
    load up splines/images into RisWidget for editing and saving
    '''

    def __init__(self, rw, spline_dir, work_dir, pca):
        self.rw = rw
        split_view.split_view_rw(self.rw)
        self.spline_edit = spline_utils.SplineEdit(rw, pca)
        spline_field = annotator.OverlayAnnotation('tck', self.spline_edit)
        self.rw.add_annotator(fields=[spline_field])
        self.spline_load = spline_utils.SplineLoad(rw, spline_dir, work_dir)
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
            spline_dict[timepoint]=page.annotations['tck']

        with open(export_file, 'wb') as save_file: pickle.dump(spline_dict, save_file)
        save_file.close()




