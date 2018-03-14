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

import spline_editor
import spline_generator
import skeleton
import straighten_worms

def generate_splines(mask_dir, export_dir):
    '''First step in the pipeline that generates splines
    from masks
    '''
    spline_generator.generate_splines(mask_dir, export_dir)

class SplineAnnotator:
'''Class that will load in worms and their splines and allow you to alter splines
and save them out.
'''
    def __init__(self, ris_widget, spline_path, image_dir, pca_stuff):
        '''Initialize the SplineAnnotator object

        Parameters
        ----------------
        ris_widget: ris_widget object

        spline_path: path to the spline_dict 
        '''
        self.rw = ris_widget
        self.spline_dict = pickle.load(open(spline_path, 'rb'))
        self.image_dir = pathlib.Path(image_dir)
        mean, pcs, norm_pcs, variances, total_variance, positions, norm_positions = pca_stuff
        self.pca_stuff = pca_stuff
        self.avg_width_positions = pca.pca_reconstruct([0,0,0], pcs[:3], mean)

        #make free_spline
        split_view.split_view_rw(self.rw)
        self.fs = free_spline.FreeSpline(self.rw)
        


