import pathlib
import pickle
import numpy

import celiagg
import freeimage
from zplib.curve import spline_geometry
from zplib.curve import interpolate
from zplib.image import resample

AGG_STATE = celiagg.GraphicsState(anti_aliased=False)
AGG_PAINT = celiagg.SolidPaint(1,1,1)
AGG_TRANSFORM = celiagg.Transform()
AGG_TRANSPARENT = celiagg.SolidPaint(0,0,0,0)

def write_xy_positions(filename, positions):
    position_lines = ['{}\t{}'.format(x, y) for x, y in positions]
    with open(filename, 'w') as f:
        f.write('\n'.join(position_lines))

def save_centerline(metadata, centerline_file):
    spine_tck = metadata['spine_tck']
    positions = interpolate.spline_interpolate(spine_tck, 50)
    write_xy_positions(centerline_file, positions)

def save_landmarks(metadata, landmark_file):
    spine_tck = metadata['spine_tck']
    vulva_t = metadata['vulva_t']
    tail_t = spine_tck[0][-1]
    t_values = [0, vulva_t, tail_t]
    positions = interpolate.spline_evaluate(spine_tck, t_values)
    write_xy_positions(landmark_file, positions)

def make_mask(metadata, mask_file):
    spine_tck = metadata['spine_tck']
    width_tck = metadata['width_tck']
    outline = spline_geometry.outline(spine_tck, width_tck, num_points=400)[-1]
    image = numpy.zeros((1040, 1388), dtype=numpy.uint8) # Celiagg convention: image.shape = (H, W)
    canvas = celiagg.CanvasG8(image)
    path = celiagg.Path()
    path.lines(outline)
    path.close()
    canvas.draw_shape(path, AGG_TRANSFORM, AGG_STATE, fill=AGG_PAINT, stroke=AGG_TRANSPARENT)
    freeimage.write(image.T, mask_file) # freeimage convention: image.shape = (W, H). So take transpose.

def make_warp(metadata, image_file, warp_file):
    image = freeimage.read(image_file)
    spine_tck = metadata['spine_tck']
    width_tck = metadata['width_tck']
    widths = interpolate.spline_interpolate(width_tck, 100)
    warp_width = 2 * widths.max() # the worm widths above are from center to edge, so twice that is edge-to-edge
    warped = resample.sample_image_along_spline(image, spine_tck, warp_width)
    mask = resample.make_mask_for_sampled_spline(warped.shape[0], warped.shape[1], width_tck)
    warped[~mask] = 0
    freeimage.write(warped, warp_file) # freeimage convention: image.shape = (W, H). So take transpose.

def process_masks_and_warps(data_dir):
    data_dir = pathlib.Path(data_dir)
    metadata_dir = data_dir / 'metadata'
    image_dir = data_dir / 'images'
    mask_dir = data_dir / 'masks'
    mask_dir.mkdir(exist_ok=True)
    warp_dir = data_dir / 'warps'
    warp_dir.mkdir(exist_ok=True)
    for metadata_file in metadata_dir.glob('*.pickle'):
        name = metadata_file.stem
        print(name)
        metadata = pickle.load(metadata_file.open('rb'))
        image_file = image_dir / (name + '.png')
        make_mask(metadata, mask_dir / (name + '.png'))
        make_warp(metadata, image_file, warp_dir / (name + '.png'))

if __name__ == '__main__':
    process_masks_and_warps('worm_data')