import pathlib
import pickle
import numpy as np

import freeimage
from zplib.curve import spline_geometry
from zplib.curve import interpolate
from zplib.image import resample
import skeleton


def warp_image(spine_tck, width_tck, image, warp_file):

    #image = freeimage.read(image_file)

    #730 was determined for the number of image samples to take perpendicular to the spine
    #from average length of a worm (as determined from previous tests)
    warped = resample.warp_image_to_standard_width(image, spine_tck, width_tck, width_tck, 730)
    #warped = resample.sample_image_along_spline(image, spine_tck, warp_width)
    mask = resample.make_mask_for_sampled_spline(warped.shape[0], warped.shape[1], width_tck)
    warped[~mask] = 0
    print("writing unit worm to :"+str(warp_file))
    freeimage.write(warped, warp_file) # freeimage convention: image.shape = (W, H). So take transpose.

def get_bf_image(image_file):
    """Easy way to find the image file
    Assumed image_dir is in the format: image_dir/worm_run/timestamp bf.png
    Assume image_file is in the format: image_dir/wormm_run/timestamp hmask.png
    """
    #im_dir = pathlib.Path(image_dir)
    timestamp = image_file.stem.split(" ")[0]
    worm_id = image_file.parent

    return worm_id.joinpath(timestamp+" bf.png")

def straighten_worms_from_rw(rw, warp_path):
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
    tck = skeleton.generate_splines(traceback, dist)
    #print("tck length: ",len(tck))
    width_tck = skeleton.width_spline(traceback, dist)

    warp_image(tck, width_tck, crop_img, warp_dir.joinpath(warp_name))
