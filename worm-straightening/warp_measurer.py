'''Use this to straighten bf and flourescent worm images 
from masks and get some statistics about various flourescent
expression. 
'''
import pathlib
import pickle
import numpy as np
from scipy import ndimage

import freeimage
from zplib.curve import spline_geometry
from zplib.curve import interpolate
from zplib.image import resample
from PyQt5 import Qt, QtGui
from ris_widget import ris_widget
import skeleton
import spline_editor
import straighten_worms

def parse_inputs(img_dir, keywords=['*bf.png', '*mask.png']): 
    worm_positions=[]
    compiled_images={file_glob:[] for file_glob in keywords}
    for subdir in list(img_dir.iterdir()): 
        if subdir.is_dir():
            r=re.search('\d{1,3}[/]?$', str(subdir))    # For post processed images....
            worm_positions.append(('/'+r.group()))           
            for file_glob in compiled_images.keys():
            #~ if 'mask' in file_glob:
                #~ compiled_images[file_glob].append(subdir.glob(file_glob))
                
            #~ else:
                #~ compiled_images[file_glob].append((self.expt_dir/subdir.parts[-1]).glob(file_glob))
                compiled_images[file_glob].append(sorted(subdir.glob(file_glob)))   # Assuming bf images in same directory
    #compiled_images=sorted(compiled_images)        
    print('finished parsing inputs')
    return compiled_images, worm_positions

def load_images(rw, img_dir, keywords=['*bf.png', '*mask.png']):
    '''Load images/masks into RisWidget in a way that 
    can be efficiently used to warp worms/find splines.
    NOTE: Assume the images with the keywords are in the same directory
    NOTE: The way the imgs are assumed to have names in the form:
        TIMEPOINT DESCRIPTOR.png (ie. 2017-09-28t1130 mask.png)

    Parameters:
    ------------
    
    Returns:
    -----------
    '''
    img_dir = pathlib.Path(img_dir)
    for img_path in sorted(img_dir.glob(keywords[0])):
        image_stack = []
        timepoint = img_path.stem.split(' ')[0]
        for key in keywords:
            img = list(img_dir.glob(timepoint+key))[0]
            img = freeimage.read(img)
            image_stack.append(img)

        rw.flipbook_pages.append(image_stack)


def parse_inputs(self): 
    worm_positions=[]
    compiled_images={file_glob:[] for file_glob in self.file_globs}
    for subdir in list(self.work_dir.iterdir()): 
        if subdir.is_dir():
            r=re.search('\d{1,3}[/]?$', str(subdir))    # For post processed images....
            worm_positions.append(('/'+r.group()))           
            for file_glob in compiled_images.keys():
            #~ if 'mask' in file_glob:
                #~ compiled_images[file_glob].append(subdir.glob(file_glob))
                
            #~ else:
                #~ compiled_images[file_glob].append((self.expt_dir/subdir.parts[-1]).glob(file_glob))
                compiled_images[file_glob].append(sorted(subdir.glob(file_glob)))   # Assuming bf images in same directory
    #compiled_images=sorted(compiled_images)        
    print('finished parsing inputs')
    return compiled_images, worm_positions