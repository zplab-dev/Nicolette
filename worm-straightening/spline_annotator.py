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
from scipy import ndimage

import spline_editor
import spline_generator
import skeleton
import straighten_worms


class SplineAnnotator:

    def __init__(self, expt_dir, image_glob, autosave=True, start_idx=0, stop_idx=None):
        '''initialize the splineAnnotator object

        expt_dir is the experiment directory
        image_glob is the images you want to grab TODO: Get rid of this?
        '''

        self.rw = ris_widget.RisWidget()
        self.autosave = autosave

        #get all the important paths
        self.expt_dir = pathlib.Path(expt_dir)
        self.work_dir = list(self.expt_dir.glob('work_dir'))[0]
        print("Work dir found: "+str(self.work_dir))
        self.spline_dir = self.get_spline_dir()
        self.worm_ids = self.get_worm_ids()
        print("Found %i worms" % (len(self.worm_ids)))
        self.image_glob = image_glob

        self.current_index = start_idx
        self.current_worm_id = self.worm_ids[self.current_index]

        #load first worm
        self.load_worm(self.current_worm_id)

        #set up spline_viewer and spline_editor
        self.spline_view = skeleton.Spline_View(self.rw)
        self.spline_edit = spline_editor.Spline_Editor(self.rw)

        #set up key thingies
        self.actions = []
        self._add_action('prev', Qt.Qt.Key_BracketLeft, lambda: self.load_next_worm(self.current_index,-1))    # Changed these because I tended to accidentally hit side keys
        self._add_action('next', Qt.Qt.Key_BracketRight, lambda: self.load_next_worm(self.current_index,1))
        self._add_action('Save Annotations', QtGui.QKeySequence('Ctrl+S'),self.save_annotations)
        #self.rw.qt_object.main_view_toolbar.addAction(self.actions[-1])


    def _add_action(self, name, key, function):
        '''Used to add functionality to hotkeys
        '''
        action = Qt.QAction(name, self.rw.qt_object)
        action.setShortcut(key)
        self.rw.qt_object.addAction(action)
        action.triggered.connect(function)
        self.actions.append(action)


    def get_spline_dir(self):
        '''Get the path where the splines will be saved
        If the splines have already been generated then use that
        as the spline_dir, if the splines haven't been generated before
        then create the spline_dir
        '''

        if len(list(self.expt_dir.glob('spline_tcks')))==0:
            spline_dir = self.expt_dir.joinpath('spline_tcks')
            print("Making spline directory at: " + str(spline_dir))
            spline_dir.mkdir()

        return list(self.expt_dir.glob('spline_tcks'))[0]

    def get_worm_ids(self):
        '''From experiment directory, get all the worm #'s'
        '''
        subdirectories = list(self.expt_dir.glob('[0-9]*/'))
        worm_ids = []
        for path in subdirectories:
            worm_ids.append(path.name)

        return worm_ids


    def load_worm(self, worm_id):
        '''load images from one worm
        NOTE: will use masks and bf_imgs from work_dir
        '''
        print("loading worm: "+str(worm_id))
        masks = self.work_dir.glob("*"+worm_id+"*/*mask.png")

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
            self.rw.flipbook_pages.append([crop_img,img, m, skel, centerline, center_dist])
            self.rw.flipbook_pages[-1].spline_data = tb
            self.rw.flipbook_pages[-1].dist_data = center_dist
            self.rw.flipbook_pages[-1].img_path = [i, sx, sy]
            self.rw.flipbook_pages[-1].med_axis = med_axis
            self.rw.flipbook_pages[-1].name = str(i)



    def load_next_worm(self, index, offset):
        '''Load next worm in the experiment
        '''

        #save current worm splines to a pickle file
        if self.autosave==True:
            spline_editor.save_splines_from_rw(self.rw, self.spline_dir)

        # do nothing if trying to go out of bounds
        if index+offset not in range(len(self.worm_ids)): 
            return

        #clear risWidget object and load next worm
        if(len(self.rw.flipbook.pages)>0): 
            self.rw.flipbook.pages.clear()
        if self.worm_ids[index+offset]:
            self.load_worm(self.worm_ids[index+offset])

        #update indices and stuff
        self.current_worm_id = self.worm_ids[index+offset]
        self.current_index = index+offset

    def save_annotations(self):
        '''Save splines
        '''
        spline_editor.save_splines_from_rw(self.rw, self.spline_dir)