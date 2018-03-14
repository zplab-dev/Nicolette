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
import straighten_worms
import _pickle as pickle

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
        print(self.points)
        skeleton.update_spline_from_points_in_rw(self.rw, self.points)
        self.points = []
        print(self.points)
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


def save_splines_from_rw(rw, output_dir):
    '''When splines are nicely edited in RisWidget, save them to a pickle file
    '''
    warped_dir = pathlib.Path(output_dir)
    spline_dict = {}
    worm_id = None

    for i in range(len(rw.flipbook.pages)):
        #current_idx = rw.flipbook.current_page_idx

        img, sx, sy = rw.flipbook.pages[i].img_path
        traceback = rw.flipbook.pages[i].spline_data
        dist = rw.flipbook.pages[i].dist_data
        #print("worm length: ",len(traceback))
        tck = skeleton.center_spline(traceback, dist)
        #print("tck length: ",len(tck))
        width_tck = skeleton.width_spline(traceback, dist)
        #need to adjust center tck to give us the correct x,y points in the bf
        tck[1].T[0]+=max(0, sx.start-50)
        tck[1].T[1]+=max(0, sy.start-50)

        #save tck's to a dict
        spline_dict[img] = (tck, width_tck)
        worm_id = img.parent.name
        '''
        bf_img_file = straighten_worms.get_bf_image(img)
        bf_img = freeimage.read(bf_img_file)

        warped_img_path = warped_dir.joinpath(bf_img_file.stem.split(" ")[0] + "_warped.png")
        img_name = bf_img_file.stem + "_warped.png"
        straighten_worms.warp_image(tck, width_tck, bf_img, warped_img_path)'''

    #save spline_dict out
    output_name = warped_dir.joinpath(worm_id+'_tck.p')
    print("Pickling spline_dict and saving to: " + str(output_name))

    pickle.dump(spline_dict, open(output_name, 'wb'))
