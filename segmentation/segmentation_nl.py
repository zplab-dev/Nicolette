import pathlib
import os
import subprocess
import sys
import platform
import json
import pickle
import multiprocessing
import functools
import time

import numpy as np
from scipy.ndimage import morphology
from skimage import measure


import freeimage
from elegant import load_data
from zplib.image import mask


# Assume that this module is in the root/base directory of segmentation 
# code and compiled executables are in the 'matlab_compiled' subfolder
#base_dir = pathlib.Path(__file__).parent

base_dir = pathlib.Path('/mnt/lugia_array/Sinha_Drew/lrr_wormseg_cuda9/worm')  # WILL EDIT

# Modify the LD_LIBRARY_PATH env variable to allow the segmentation code
# to see the necessary MATLAB runtime environment
exe_env = os.environ.copy()
if 'centos' in platform.platform(): # On the cluster
    mcr_root = '/export/matlab/MCR/R2018a'
elif 'Darwin' in platform.platform():
    mcr_root = '/Applications/MATLAB/MATLAB_Runtime'
else:
    mcr_root = '/usr/local/MATLAB/MATLAB_Runtime'

addl_paths = [mcr_root + '/v94/runtime/glnxa64',
    mcr_root + '/v94/bin/glnxa64',
    mcr_root + '/v94/sys/os/glnxa64',
    mcr_root + '/v94/extern/bin/glnx64']
addl_paths = ':'.join(addl_paths)
exe_env["LD_LIBRARY_PATH"] = exe_env.get("LD_LIBRARY_PATH",'') + ':' + addl_paths

# Put additional shared library files on the load/preload path if on the cluster
if 'centos' in platform.platform():
    exe_env["LD_LIBRARY_PATH"] = exe_env["LD_LIBRARY_PATH"] + ':' + '/home/sinhad/lib'
    exe_env["LD_PRELOAD"] = exe_env.get("LD_PRELOAD",'') + ':' + '/home/sinhad/lib/glibc/build/libc.so.6'

DEFAULT_MODEL = 'data/worm_first_try/4x-dp-0/net-epoch-100.mat' #  CF's trained classifier

def process_image(image_file,
    save_directory=None, remake_masks=True, enable_gpu = False, model_path=None):
    """Process a single image and save it to disk using the separate LRR
        executable
    
    """
    image_file = pathlib.Path(image_file)
    
    if save_directory is None:
        save_directory = image_file.parent
    
    if model_path is None:
        model_path = base_dir / 'data' / 'worm_first_try' / '4x-dp-0/net-epoch-100.mat' # Point to CF's trained classifier

    if remake_masks or not (image_path.parent / (image_path.stem + ' mask' + image_path.suffix)).exists():
        subprocess.run(
            [str(base_dir / 'matlab_compiled/processImage/for_redistribution_files_only/processImage'),
                str(image_file),
                str(save_directory),
                str(int(enabled_gpu)),
                str(model_path)],
            env=exe_env)

def process_position_directory(position_directory, channels_to_process, 
    save_directory=None, remake_masks=False,
    enable_gpu = False, model_path=None):
    """Process a position directory by starting a server for the segmenter
        and feeding it images to process on the fly.
        
        Parameters
            position_directory - str/pathlib.Path to position directory
            channels_to_process - list of str image identifiers to process 
                into masks;
            save_directory - optional str/pathlib.Path where masks will be
                saved
            remake_masks - optional bool flag to denote whether the segmenter
                should remake masks for images that have already been processed
            enable_gpu - optional bool flag to turn on GPU compatibility;
                as of matconvnet-1.0-beta25, there's a bug in matconvnet
                such that mex code compiled with gpu support doesn't work
                when the computer doesn't have a gpu; thus, for cluster computing
                and working on other computers, set this to False to use
                the alternate build of the segmentation code
        
        TODO: 
            Refactor to use load_data.scan_experiment_dir??
            Clean up save_directory business - should this even be exposed?
    """
    
    position_directory = pathlib.Path(position_directory)
    
    if save_directory is None:
        experiment_directory = position_directory.parent
        position = position_directory.name
        
        save_directory = experiment_directory / 'derived_data' / 'mask' / position
    else:
        save_directory = pathlib.Path(save_directory)

    # Make sure that the save directory exists
    try:
        save_directory.mkdir()
    except FileExistsError:
        pass
    
    if model_path is None:
        model_path = base_dir / DEFAULT_MODEL

    if not enable_gpu:
        # TODO force no gpu #if 'centos' in platform.platform()?
        build_name = 'processImageBatch_nogpu'
    else:
        build_name = 'processImageBatch'
    exe_path = (base_dir / f'matlab_compiled/{build_name}/for_redistribution_files_only/{build_name}')

    try: position_annotations = load_data.read_annotation_file(
        position_directory.parent / 'annotations' / 
        (position_directory.stem + '.pickle'))
    except:
        print(position_directory.stem,' does not have annotation')
        return
    
    # Enumerate files to process 
    files_to_process = []
    for image_file in sorted(list(position_directory.iterdir())):
        if image_file.suffix[1:] in ['png','tiff']:
            MASK_FILE_EXISTS = (save_directory / image_file.name).exists()
            IS_GOOD_CHANNEL = image_file.stem.split(' ')[-1] in channels_to_process
            #IS_ALIVE = position_annotations[1][image_file.stem.split(' ')[0]]['stage'] != 'dead'
            #IS_ADULT = position_annotations[1][image_file.stem.split(' ')[0]]['stage'] == 'adult'
            
            #if (remake_masks or not MASK_FILE_EXISTS) and IS_GOOD_CHANNEL and IS_ADULT:
            if (remake_masks or not MASK_FILE_EXISTS) and IS_GOOD_CHANNEL:
                files_to_process.append(image_file)
    print("Files to process: ", files_to_process[0]) 
    if len(files_to_process) == 0:
        print(f'No files to process for position {position_directory.name}')
        return
    
    # Write out tempfile
    with (position_directory / 'image_manifest.txt').open('w') as manifest_file:
        for my_file in files_to_process:
            manifest_file.write(str(my_file)+'\n')
            manifest_file.write(str(save_directory / my_file.name)+'\n')
    
    print('Starting batch segmentation')
    subprocess.run(
        [str(exe_path),
            str(position_directory / 'image_manifest.txt'), 
            str(int(enable_gpu)),
            str(model_path)],
        env=exe_env)
    
    # Post-process written images to get just the largest component
    with (position_directory / 'image_manifest.txt').open('r') as mf_stream:
        mask_files = map(str.rstrip, mf_stream.readlines()[1::2])
    for mask_file in mask_files:
        if pathlib.Path(mask_file).exists():
            mask_image = freeimage.read(mask_file) > 0
            new_mask = mask.get_largest_object(mask_image).astype(np.uint8)
            freeimage.write(new_mask*255, mask_file)
        else:
            print(f'No mask region found for {mask_file}')
    
    # Clean up
    #(position_directory / 'image_manifest.txt').unlink()

def process_experiment(experiment_directory, channels_to_process, num_workers=None, position_filter=None, **process_args):
    experiment_directory = pathlib.Path(experiment_directory)
    
    if num_workers == None:
        num_workers = multiprocessing.cpu_count() - 2
    elif num_workers > multiprocessing.cpu_count():
        raise RuntimeError('Attempted to run jobs with more workers than cpu\'s!')
    
    if position_filter is None:
        position_filter = load_data.filter_good_complete
    
    # Make super_vignette if needed.
    if not (experiment_directory / 'super_vignette.pickle').exists():
        make_super_vignette(experiment_directory)
    
    # Make appropriate subdirectories
    if not (experiment_directory / 'derived_data').exists():
        (experiment_directory / 'derived_data').mkdir()
    if not (experiment_directory / 'derived_data' / 'mask').exists():
        (experiment_directory / 'derived_data' / 'mask').mkdir()
    
    # Enumerate position directories
    positions = load_data.read_annotations(experiment_directory)
    positions = load_data.filter_annotations(positions, position_filter=position_filter)
    print('Processing the following positions:')
    print(list(positions.keys()).__repr__())
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        try:
            bob = pool.map(
                functools.partial(process_position_directory, channels_to_process=channels_to_process, **process_args),
                [experiment_directory / pos for pos in positions])
            pool.close()
            pool.join()
            print('Terminated successfully')
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            raise

if __name__ == "__main__":

    #exp_dir = '/mnt/9karray/Pittman_Will/pipeline_test/'

    #save_dir = '/mnt/9karray/Pittman_Will/pipeline_test/cnn_masks/'

    exp_dir = '/mnt/lugia_array/20170919_lin-04_GFP_spe-9/work_dir/'

    save_dir = '/mnt/lugia_array/20170919_lin-04_GFP_spe-9/work_dir/derived_data/mask/'

    os.makedirs(save_dir, exist_ok = True)


    t0 = time.time()

    for folder in os.listdir(exp_dir):

        if '20170919_lin-04_GFP_spe-9' in folder:

            process_position_directory(exp_dir+folder, ['bf'],save_directory=save_dir+folder, enable_gpu=False, model_path='/mnt/lugia_array/segmentation_code/worm_segmentor_1/worm_segmenter/models/default_CF.mat')
            #process_position_directory(exp_dir+folder, ['bf'],save_directory=save_dir+folder, enable_gpu=True, model_path=None)
            #except: print('No annotations for worm: ', item)

    print('This experiment took: ' + str(time.time() - t0) + ' seconds.\n')