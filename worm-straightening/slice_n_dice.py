import numpy as np

import freeimage

from elegant import worm_spline
from zplib.curve import spline_geometry
from zplib.curve import interpolate
from zplib.image import colorize


def calculate_keypoints(keypoints, center_tck):
    """In order for longitudinal_warp_spline to work we
    need the keypoint positions to be as a ratio of the 
    length of the center_tck.

    Parameters:
        keypoints: dictionary from the annotations file
        center_tck: tck of the centerline of the worm
    """

    length = spline_geometry.arc_length(center_tck)
    x, y = zip(*list(keypoints.values()))
    return x/length


def output_aligned_image(image, center_tck, width_tck, keypoints, output_file, t_out = [0.07, 0.15, 0.5, 0.95]):
    """ Make warped pictures and align the keypoints
    """
    #get the alignments for longitudinal_warp_spline
    t_in = calculate_keypoints(keypoints, center_tck)
    aligned_center, aligned_width = worm_spline.longitudinal_warp_spline(t_in, t_out, center_tck, width_tck=width_tck)

    #output the image
    warps = warps = worm_spline.to_worm_frame(image, aligned_center, width_tck=aligned_width, width_margin=0, standard_width=aligned_width)
    mask = worm_spline.worm_frame_mask(aligned_width, warps.shape)

    #change warps to an 8-bit image
    bit_warp = colorize.scale(warps).astype('uint8')
    #make an rgba image, so that the worm mask is applied
    rgba = np.dstack([bit_warp, bit_warp, bit_warp, mask])
    #save the image
    freeimage.write(rgba, output_file)