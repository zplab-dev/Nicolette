from ris_widget.ris_widget import RisWidget
from PyQt5 import Qt, QtGui
import pathlib
import numpy as np
import freeimage
import gc

import zplib.image.mask as zplib_image_mask
import skeleton

class Mask_editor:
    '''Edit the mask for the skeleton stuff
    '''
    def __init__(self, rw):
        '''
        '''
        self.rw = rw
        self.layout = Qt.QFormLayout()
        mask_widget = Qt.QWidget()
        mask_widget.setLayout(self.layout)

        self.edit = Qt.QPushButton("Edit Mask")
        self.layout.addRow(self.edit)
        self.rw.flipbook.layout().addWidget(mask_widget)

        self.edit.clicked.connect(self._on_edit_clicked)

        self.editing = False


    def _on_edit_clicked(self):
        if self.editing:
            self.stop_editing()
        else:
            self.start_editing()
            
    def start_editing(self):        
        #Do everything in second layer; will need to modify Willie's code for worm measurement
        self.editing = True
        self.edit.setText('Save Edits')
        
        # Bring focus to mask layer
        sm = self.rw.qt_object.layer_stack._selection_model
        m = sm.model()
        sm.setCurrentIndex(m.index(0,0),
            Qt.QItemSelectionModel.SelectCurrent|Qt.QItemSelectionModel.Rows)
        
        self.rw.layers[2].opacity = 0.5
        self.rw.layers[2].tint = (1.0,0,0,1.0)
        self.rw.qt_object.layer_stack_painter_dock_widget.show()
        self.rw.qt_object.layer_stack_painter.brush_size_lse.value = 13
        #current_page_idx = self.rw.flipbook.current_page_idx
        #current_page = self.rw.flipbook.pages[current_page_idx]
        #self.current_page[1].set(data=(self.current_page[1].data*0).astype('bool'))
        
        
    def stop_editing(self,save_work=True):
        # Convert outline to mask 
        outline = zplib_image_mask.get_largest_object(
            self.rw.layers[2].image.data>0)
        new_mask = zplib_image_mask.fill_small_area_holes(outline,300000).astype('uint8')
        new_mask = new_mask>0
        #new_mask[new_mask>0] = -1
        self.rw.layers[2].image.set(data=(new_mask>0).astype('bool'))
        self.rw.layers[2].tint = (1.0,1.0,1.0,1.0)

        #update spline data
        m, skel, centerline, center_dist, med_axis, tb = skeleton.skel_and_centerline(new_mask)
        current_page_idx = self.rw.flipbook.current_page_idx
        self.rw.layers[1:]=[skel, centerline, center_dist]
        self.rw.flipbook.pages[current_page_idx].spline_data = tb
        self.rw.flipbook.pages[current_page_idx].dist_data = center_dist
        self.rw.flipbook.pages[current_page_idx].med_axis = med_axis

        
        self.rw.qt_object.layer_stack_painter_dock_widget.hide()
        
        """if self.current_page[1].data.any():
                                    if not pathlib.Path(str(self.working_file).replace('mask.png','mask_oldwf.png')).exists():
                                        self.working_file.rename(str(self.working_file).replace('mask.png','mask_oldwf.png'))
                                    freeimage.write(new_mask, self.working_file)"""
        
        self.editing = False
        #self.working_file = None    # Do this after saving out since need the working file above
        self.edit.setText('Edit Mask')