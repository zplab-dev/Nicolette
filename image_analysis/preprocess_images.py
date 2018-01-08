import numpy
import pathlib
import freeimage
import scipy.ndimage as ndimage
from skimage import transform
from zplib.image import colorize
from zplib.image import pyramid

def mode_normalize(image):
    '''normalize image by dividing by the non-zero mode
    '''
    m = numpy.bincount(image[image>0]).argmax()
    return image.astype(numpy.float32) / m

basic_bf_scaling = dict(min=0, max=1.09, gamma=0.75)

def write_mode_normalized(in_file, out_file, shrink=2, min=0, max=1.09, gamma=0.75):
    image = freeimage.read(in_file)
    #small_image = pyramid.pyr_down(image, shrink).astype(numpy.uint16)
    cropped_image = crop_img(image)
    small_image = pyr_down_set(cropped_image, (768,640)).astype(numpy.uint16)
    norm_image = mode_normalize(small_image)
    colorize.write_scaled(norm_image, out_file, min, max, gamma)

def pyr_down_set(image, shape, downscale=2, nyquist_attenuation=0.01):
    """Return an image downsampled by the requested factor.

    Parameters:
        image: numpy array
        shape: dimensions you want for the output image (width, height).
        nyquist_attenuation: controls strength of low-pass filtering (see
            documentation for downsample_sigma() for detailed description).
            Larger values = more image blurring.

    Returns: image of type float32
    """
    out_shape = shape
    downscale=numpy.divide(image.shape,out_shape).max().astype(int)
    print("downscale factor: "+str(downscale))
    sigma = downsample_sigma(downscale, nyquist_attenuation)
    smoothed = ndimage.gaussian_filter(image.astype(numpy.float32), sigma, mode='reflect')
    return transform.resize(smoothed, out_shape, order=1, mode='reflect', preserve_range=True)

def downsample_sigma(scale_factor, nyquist_attenuation=0.05):
    """Calculate sigma for gaussian blur that will attenuate the nyquist frequency
    of an image (after down-scaling) by the specified fraction. Surprisingly,
    attenuating by only 5-10% is generally sufficient (nyquist_attenuation=0.05
    to 0.1).
    See http://www.evoid.de/page/the-caveats-of-image-down-sampling/ .
    """
    return scale_factor * (-8*numpy.log(1-nyquist_attenuation))**0.5

def write_mask(in_file, out_file, shape=(768,640)):
    """Deals with worm masks (hmasks, etc.) since you
    don't need to mode normalize them
    """
    image = freeimage.read(in_file)
    cropped_image = crop_img(image)
    #small_image = pyramid.pyr_down(image, shrink).astype(numpy.uint16)
    small_image = (pyr_down_set(image, (768,640)).astype(numpy.uint16) > 128).astype(numpy.uint8)*255
    freeimage.write(small_image, out_file)

def crop_img(img):
    '''Crop image to get rid of the weird black border thing
    '''
    return img[234:2433, 277:1832]

def preprocess(image_dir, out_dir, image_type="bf"):
    '''preprocess all imamges in an image directory
    Assumes the masks are in other folders in the directory you provide it
    This is the format that I have used in the images I've been using 

    image_type allows you to  specifiy which images you want to normalize
    '''

    img_dir=pathlib.Path(image_dir)
    out_dir=pathlib.Path(out_dir)

    if image_type=="mask":
        #need to preprocess masks differently from regular bf files
        #Don't need to mode normalize the masks
        for i in list(img_dir.glob('**/*mask*.png')):
            #get subfolder
            print(i.name)
            subfolder=i.parts[-2]
            name=i.name
            out_folder=out_dir.joinpath(subfolder)
            #make sure out folder exists,
            #if not create it
            if not out_folder.exists():
                out_folder.mkdir()
                #print(str(out_folder))

            out_file=out_folder.joinpath(name)
            print('Saving file to: '+str(out_file))
            write_mask(i, out_file)
    else:
        for i in list(img_dir.glob('**/*bf.png')):
            #get subfolder
            print(i.name)
            subfolder=i.parts[-2]
            name=i.name
            out_folder=out_dir.joinpath(subfolder)
            #make sure out folder exists,
            #if not create it
            if not out_folder.exists():
                out_folder.mkdir()
                #print(str(out_folder))

            out_file=out_folder.joinpath(name)
            print('Saving file to: '+str(out_file))
            write_mode_normalized(i, out_file)

