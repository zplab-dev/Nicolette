import pathlib
import pickle
import numpy as np

import freeimage
from zplib.curve import spline_geometry
from zplib.curve import interpolate
from zplib.image import resample

meta_dir=pathlib.Path('/media/nicolette/C4FC-ED48/worm_data/metadata')
image_dir=pathlib.Path('/media/nicolette/C4FC-ED48/worm_data/images/')
warp_dir=pathlib.Path("/media/nicolette/C4FC-ED48/worm_data/warps/")
meta_files=list(meta_dir.glob('*pickle'))


def get_avg_widths(metadata_list):
    """Get the average widths of worms 3-7 days old to 
    to make a unit worm from. 
    """
    #average the widths at 100 points along the spline
    widths=[]
    for m in metadata_list.values():
        width_tck=m['width_tck']
        widths.append(interpolate.spline_interpolate(width_tck, 100))

    widths=np.array(widths)
    widths_avg=np.mean(widths, axis=0)

    return widths_avg

def get_avg_lengths(metadata_list):
    """Get the average lengths of worms 3-7 days old to
    make a unit worm from
    """
    lengths=[]
    for m in metadata_list.values():
        spine_tck=m['spine_tck']
        lengths.append(spline_geometry.arc_length(spine_tck))

    lengths=np.array(lengths)
    lengths_avg=np.mean(lengths, axis=0)
    return lengths_avg

def extract_metadata(meta_files):
    """Get metadata for worms between 3-7 days old
    Return a dictionary with the worm identifier/name and metadata (worm_name, metadata)
    """
    #get metadata for worms between 3-7 days old
    metadata_list={} 
    for m in meta_files:
        meta=pickle.load(m.open('rb'))
        if meta['age_days']>3.0 and meta['age_days']<8.0:
            metadata_list[m.stem]=meta

    return metadata_list

def make_unit_worm(metadata, avg_widths, avg_length, image_file, warp_file):
    """Make a unit worm and save the image to the warp_file

        Parameters:
        metadata: dictionary
        avg_widths: array, size (n_points,)
            array of average widths with which to make the worm
        image_file: str;
            what file to use to subsample the textures from 
        warp_file: str;
            where to store the new warp file
    """
    image = freeimage.read(image_file)
    spine_tck = metadata['spine_tck']
    width_tck = metadata['width_tck']
    warp_width = 2 * avg_widths.max() # the worm widths above are from center to edge, so twice that is edge-to-edge
    avg_widths_tck=interpolate.fit_nonparametric_spline(np.linspace(0,1,100),avg_widths)

    warped = resample.warp_image_to_standard_width(image, spine_tck, width_tck, avg_widths_tck, warp_width, length=avg_length)
    #warped = resample.sample_image_along_spline(image, spine_tck, warp_width)
    mask = resample.make_mask_for_sampled_spline(warped.shape[0], warped.shape[1], width_tck)
    warped[~mask] = 0
    print("writing unit worm to :"+str(warp_file))
    freeimage.write(warped, warp_file) # freeimage convention: image.shape = (W, H). So take transpose.


def unit_worm(metadata_dir, image_dir, warp_dir):
    """Makes unit worm files/images for a directory of worms
    
    Parameters:
        metadata_dir: str, directory where the metadata is stored
        image_dir: str, directory where individual images of worms are stored
        warp_dir: str, directory where you want to store the unit worm images

    Returns:
        Saves unit worm images into the warp_dir
    """
    meta_dir=pathlib.Path(metadata_dir)
    image_dir=pathlib.Path(image_dir)
    unit_worm_dir = pathlib.Path(warp_dir)
    meta_files=list(meta_dir.glob('*pickle'))
    metadata_list=extract_metadata(meta_files)
    avg_widths=get_avg_widths(metadata_list)
    avg_length=get_avg_lengths(metadata_list)
    
    #make unit worms for all the worms in the metadata_dir
    for k,v in metadata_list.items():
        image_file=list(image_dir.glob(k+"*"))[0]
        out_file=unit_worm_dir.joinpath(k+"_unit_worm.png")
        make_unit_worm(v, avg_widths, avg_length, image_file, out_file)
