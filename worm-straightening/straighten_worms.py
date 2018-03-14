import pathlib
import pickle
import numpy as np
from scipy import ndimage

import freeimage
from zplib.curve import spline_geometry
from zplib.curve import interpolate
from zplib.image import resample
from zplib.image import colorize
from PyQt5 import Qt, QtGui
from ris_widget import ris_widget
import skeleton
import spline_editor


def warp_image(spine_tck, width_tck, image, warp_file):

    #image = freeimage.read(image_file)

    #730 was determined for the number of image samples to take perpendicular to the spine
    #from average length of a worm (as determined from previous tests)
    warped = resample.warp_image_to_standard_width(image, spine_tck, width_tck, width_tck, 730)
    #warped = resample.sample_image_along_spline(image, spine_tck, warp_width)
    mask = resample.make_mask_for_sampled_spline(warped.shape[0], warped.shape[1], width_tck)
    warped = colorize.scale(warped).astype('uint8')
    warped[~mask] = 255

    print("writing warped worm to :"+str(warp_file))
    #return warped
    freeimage.write(warped, warp_file) # freeimage convention: image.shape = (W, H). So take transpose.

def get_bf_image(image_file, matt=False):
    """Easy way to find the image file
    Assumed image_dir is in the format: image_dir/worm_run/timestamp bf.png
    Assume image_file is in the format: image_dir/wormm_run/timestamp hmask.png
    """
    #im_dir = pathlib.Path(image_dir)
    if matt:
        #print(image_file.stem)
        timestamp = image_file.stem.split('_')[-1]
        worm_id = image_file.parent
        bf_img = worm_id.joinpath("position_"+timestamp+".png")
    else:
        timestamp = image_file.stem.split(" ")[0]
        worm_id = image_file.parent
        bf_img = worm_id.joinpath(timestamp+" bf.png")
    return bf_img

def crop_img(rw):
    current_idx = rw.flipbook.current_page_idx
    img, sx, sy = rw.flipbook.pages[current_idx].img_path
    bf_img_file = get_bf_image(img)
    bf_img = freeimage.read(bf_img_file)
    #need to crop image like the risWidget ones
    x = slice(max(0, sx.start-50),min(bf_img.shape[0], sx.stop+50))
    y = slice(max(0, sy.start-50),min(bf_img.shape[1], sy.stop+50))
    crop_img = bf_img[x,y]


def straighten_worms_from_rw(rw, warp_path):
    '''Straighten a worm that's currently being shown in given RisWidget
    and save it in the warp_path directory.

    Parameters:
    ------------
    rw: RisWidget Object
        Reference to the RisWidget object with the worm you want to
        straighten

    warp_path: String
        Directory that you want to save your warped worm image into

    Returns:
    -----------
    '''
    warp_dir = pathlib.Path(warp_path)
    if not warp_dir.exists():
        warp_dir.mkdir()

    current_idx = rw.flipbook.current_page_idx

    img, sx, sy = rw.flipbook.pages[current_idx].img_path
    bf_img_file = get_bf_image(img)
    bf_img = freeimage.read(bf_img_file)
    #need to crop image like the risWidget ones
    x = slice(max(0, sx.start-50),min(bf_img.shape[0], sx.stop+50))
    y = slice(max(0, sy.start-50),min(bf_img.shape[1], sy.stop+50))
    crop_img = bf_img[x,y]

    warp_name = bf_img_file.stem.split(" ")[0] + "_"+str(current_idx)+"_warp.png"

    
    traceback = rw.flipbook.pages[current_idx].spline_data
    dist = rw.flipbook.pages[current_idx].dist_data
    #print("worm length: ",len(traceback))
    tck = skeleton.center_spline(traceback, dist)
    #print("tck length: ",len(tck))
    width_tck = skeleton.width_spline(traceback, dist)

    warp_image(tck, width_tck, crop_img, warp_dir.joinpath(warp_name))


def load_imgs(rw, img_dir):
    '''Load images/masks into RisWidget in a way that 
    can be efficiently used to warp worms/find splines

    Parameters:
    ------------
    
    Returns:
    -----------
    '''
    img_dir = pathlib.Path(img_dir)

    #load in masks/images
    #Note images will be cropped to be close to the mask
    for i in list(img_dir.glob("*mask.png")):
        im=freeimage.read(i)
        im=im>0
        sx, sy=ndimage.find_objects(im)[0]
        x = slice(max(0, sx.start-50),min(im.shape[0], sx.stop+50))
        y = slice(max(0, sy.start-50),min(im.shape[1], sy.stop+50))
        img = im[x,y]
        bf = freeimage.read(get_bf_image(i))
        crop_img = bf[x,y]
        m, skel, centerline, center_dist, med_axis, tb =skeleton.skel_and_centerline(img)
        #rw.flipbook_pages.append([img, m, skel, centerline, center_dist])
        rw.flipbook_pages.append([crop_img,img, m, skel, centerline, center_dist])
        rw.flipbook_pages[-1].spline_data = tb
        rw.flipbook_pages[-1].dist_data = center_dist
        rw.flipbook_pages[-1].img_path = [i, sx, sy]
        rw.flipbook_pages[-1].med_axis = med_axis

    #Add all the fancy buttons and stuff to risWidget
    spline_view = skeleton.Spline_View(rw)
    spline_edit = spline_editor.Spline_Editor(rw)
    ok = ris_widget.dock_widgets.annotator.BoolField('warp OK', default=True)
    fields = [ok]
    annotator = ris_widget.dock_widgets.Annotator(rw, fields)
    #warp_worm = WarpWorm(rw)


class WarpWorm:
    '''
    '''
    def __init__(self, rw):
        self.rw = rw
        self.rw_warp = ris_widget.RisWidget()
        self.layout = Qt.QFormLayout()
        spline_widget = Qt.QWidget()
        spline_widget.setLayout(self.layout)

        self.warp = Qt.QPushButton("Warp Worm")
        self.clear = Qt.QPushButton("Clear")
        self.layout.addRow(self.warp)
        self.rw.flipbook.layout().addWidget(spline_widget)

        self.warp.clicked.connect(self._on_warp_clicked)

    def _on_warp_clicked(self):
        #warp the image from the risWidget
        warped, img_name = self.straighten_worms()
        current_idx = self.rw.flipbook.current_page_idx
        
        #update risWidget
        self.rw_warp.flipbook.pages.append([warped])
        self.rw_warp.flipbook.pages[-1].name = img_name

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
    
    def warp_image(self, spine_tck, width_tck, image):
        warped = resample.warp_image_to_standard_width(image, spine_tck, width_tck, width_tck, 730)
        #warped = resample.sample_image_along_spline(image, spine_tck, warp_width)
        mask = resample.make_mask_for_sampled_spline(warped.shape[0], warped.shape[1], width_tck)
        warped[~mask] = 0

        return warped

    def straighten_worms(self):

        current_idx = self.rw.flipbook.current_page_idx

        img, sx, sy = self.rw.flipbook.pages[current_idx].img_path
        bf_img_file = get_bf_image(img)
        bf_img = freeimage.read(bf_img_file)
        #need to crop image like the risWidget ones
        x = slice(max(0, sx.start-50),min(bf_img.shape[0], sx.stop+50))
        y = slice(max(0, sy.start-50),min(bf_img.shape[1], sy.stop+50))
        crop_img = bf_img[x,y]

        traceback = self.rw.flipbook.pages[current_idx].spline_data
        dist = self.rw.flipbook.pages[current_idx].dist_data
        #print("worm length: ",len(traceback))
        tck = skeleton.center_spline(traceback, dist)
        #print("tck length: ",len(tck))
        width_tck = skeleton.width_spline(traceback, dist)

        warped = self.warp_image(tck, width_tck, crop_img)
        img_name = bf_img_file.stem.split(" ")[0] + "_"+str(current_idx)

        return warped, img_name
