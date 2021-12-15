import pathlib
import numpy as np
import tifffile as tiff
from read_roi import read_roi_zip, read_roi_file
from skimage import draw as skdraw


def roi2masks(roi_dict, image_shape):
    masks = np.zeros((len(roi_dict),) + image_shape, dtype=bool)
    for i, key in enumerate(roi_dict):
        roi = roi_dict[key]
        r = np.array(roi['y'])
        c = np.array(roi['x'])
        rr, cc = skdraw.polygon(r, c, image_shape)
        masks[i][rr, cc] = True

    return masks


def read_roi(roi_file, image_shape):
    roi_file = pathlib.Path(roi_file)
    roi_ext = roi_file.suffix.lower()
    if(roi_ext == '.roi'):
        roi_dict = read_roi_file(roi_file)
    elif(roi_ext == '.zip'):
        roi_dict = read_roi_zip(roi_file)
    elif(roi_ext == '.tif' or roi_ext == '.tiff'):
        masks = tiff.imread(roi_file).astype(bool)
        # If no ROI, the image has one black page, which should be removed
        if(len(masks) == 1 and not np.any(masks[0])):
            masks = np.zeros((0,) + masks.shape[1:])
        return masks
    else:
        print('file not found: %s' % roi_file)
        roi_dict = {}

    return roi2masks(roi_dict, image_shape)
