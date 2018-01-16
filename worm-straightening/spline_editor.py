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
import skeleton

class Spline_Editor:

    def __init__(self, rw):
        self.rw = rw
        self.layout = Qt.QFormLayout()
        spline_widget = Qt.QWidget()
        spline_widget.setLayout(self.layout)

        self.edit = Qt.QPushButton("Edit Spline Points")
        self.clear = Qt.QPushButton("Clear")
        self.layout.addRow(self.edit)
        self.layout.addRow(self.clear)
        self.clear.setVisible(False)
        self.rw.flipbook.layout().addWidget(spline_widget)

        self.edit.clicked.connect(self._on_edit_clicked)

        self.points = []
        self.editing = False

    def _on_edit_clicked(self):
        if self.editing:
            self.stop_editing()
        else:
            self.start_editing()

    def _on_clear_clicked(self):
        self.points = []
        print("cleared points")

    def start_editing(self):
        '''Once the Editing button is pressed allow
        people to click points
        '''
        self.editing = True
        self.edit.setText('Save Edits')
        self.rw.image_view.mouse_release.connect(self.mouse_release)
        #self.layout.addRow(self.clear)
        self.clear.setVisible(True)
        self.clear.clicked.connect(self._on_clear_clicked)

    def stop_editing(self):
        '''Once the save edits button is pressed
        save the new spline
        '''
        skeleton.update_spline_from_points_in_rw(self.rw, self.points)
        self.points = []
        self.editing = False
        self.edit.setText('Edit Spline Points')
        self.clear.setVisible(False)
        self.rw.image_view.mouse_release.disconnect(self.mouse_release)
        

    def mouse_release(self, pos, modifiers):
        x,y = pos.x(),pos.y()
        #print(modifiers)
        print("clicked point: ",x,y)
        self.points.append((int(x),int(y)))
        print(self.points)
        
