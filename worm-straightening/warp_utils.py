import collections
import pathlib
import pickle
import numpy as np
from scipy import ndimage

import freeimage
from zplib.image import resample
from zplib.image import colorize

def warp_image(spine_tck, width_tck, image_file, warp_file):
    """Warp an image of a worm to a specified place

    Parameters:
        spine_tck: parametric spline tck tuple corresponding to the centerline of the worm
        width_tck: non-parametric spline tck tuple corresponding to the widths of the worm
        image_file: path to the image to warp the worm from
        warp_file: path where the warped worm image should be saved to
    """

    image = freeimage.read(image_file)
    warp_file = pathlib.Path(warp_file)

    #730 was determined for the number of image samples to take perpendicular to the spine
    #from average length of a worm (as determined from previous tests)
    warped = resample.warp_image_to_standard_width(image, spine_tck, width_tck, width_tck, 730)
    #warped = resample.sample_image_along_spline(image, spine_tck, 730)
    mask = resample.make_mask_for_sampled_spline(warped.shape[0], warped.shape[1], width_tck)
    warped = colorize.scale(warped).astype('uint8')
    warped[~mask] = 255

    print("writing warped worm to :"+str(warp_file))
    #return warped
    if not warp_file.exists():
        warp_file.parent.mkdir(exist_ok=True)

    freeimage.write(warped, warp_file) # freeimage convention: image.shape = (W, H). So take transpose.

def warp_worms(expt_dir, positions):
    """Warp many worms at one time to a folder 'warped_images'
    NOTE: this assumes that the experiment has annotations 
    (i.e. the worms have been previously straightened)

    Assumptions are made about how the gfp images (since we want gfp images in the end)
    are located/named. In Holly's experiments, the work_dir contains altered gfp images which
    we don't necessarily want to straighten (we need the ones in the experiment root).
    Unfortunately, the worms in the work_dir are named something else than the experiment root
    (i.e. '20170919_lin-04_GFP_spe-9 000' vs '000')

    Parameters:
        expt_dir: path to an experiment directory
        positions: ordered dictionary of position names (i.e. worm names)
                mapping to ordered dictionaries of timepoint names, each of
                which maps to a list of image paths to load for each timepoint.
                Note: load_data.scan_experiment_dir() provides exactly this.
    """
    expt_dir = pathlib.Path(expt_dir)
    annotation_dir = expt_dir / 'annotations'
    warp_dir = expt_dir / 'warped_images'
    
    if not warp_dir.exists():
        warp_dir.mkdir(exist_ok = True)

    for worm, timepoints in positions.items():
        annotation_file = list(annotation_dir.glob('*'+worm+'.pickle'))[0]
        annotations = pickle.load(open(annotation_file, 'rb'))
        positions, tcks = annotations
        for tp, imgs in timepoints.items():
            center_tck, width_tck = tcks[tp]['pose']
            for img in imgs:
                warp_file = warp_dir / worm / (img.name)
                
                warp_image(center_tck, width_tck, img, warp_file)




