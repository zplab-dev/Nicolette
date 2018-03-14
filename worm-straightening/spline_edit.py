import pathlib
from zplib import datafile
from ris_widget import ris_widget
import PyQt5.Qt as Qt
import PyQt5.QtGui as QtGui
import glob
import sys
import freeimage
import numpy as np
from scipy import ndimage
from zplib.image import mask
from zplib.image import active_contour
from zplib.curve import interpolate
from zplib.curve import spline_geometry
from zplib.curve import geometry
from zplib import pca
from ris_widget.overlay import free_spline


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
        new_width_tck = self._smooth_width_from_pca(self.avg_width_tck)
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
        #old_width_tck = self._width_tck
        old_width_tck = self._width_tck
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