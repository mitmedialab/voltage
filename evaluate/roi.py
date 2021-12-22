import pathlib
import numpy as np
import tifffile as tiff
from read_roi import read_roi_zip, read_roi_file
from skimage import draw as skdraw


def roi2masks(roi_dict, image_shape):
    """
    Convert ROI contour locations into binary image masks.

    Parameters
    ----------
    roi_dict : dictionary of ROI contours
        Each element of the dictionary specifies one ROI contour, which is
        expressed by a dictionary of X and Y coordinates of the contour.
    image_shape : 2-tuple of int
        Height and width of the output masks.

    Returns
    -------
    masks : 3D numpy.ndarray of boolean
        ROI masks whose shape is (# ROIs, height, width).

    """
    masks = np.zeros((len(roi_dict),) + image_shape, dtype=bool)
    for i, key in enumerate(roi_dict):
        roi = roi_dict[key]
        r = np.array(roi['y'])
        c = np.array(roi['x'])
        rr, cc = skdraw.polygon(r, c, image_shape)
        masks[i][rr, cc] = True

    return masks


def read_roi(roi_file, image_shape):
    """
    Read an ROI file and return an array of binary masks representing the ROIs.
    The file can be .roi specifying a single ROI, .zip containing multiple of
    .roi, or a multipage binary TIFF file.
    The file type is determined based on the file name extension.

    Parameters
    ----------
    roi_file : string
        ROI file name.
    image_shape : 2-tuple of int
        Height and width of the output masks. This is used when the ROI file
        type is .roi or .zip, as these only specify ROI contour locations
        without specifying the image shape that the ROIs lie in. This value is
        not used when the ROI file type is TIFF, as TIFF specifies the shape.

    Returns
    -------
    3D numpy.ndarray of boolean
        Array of binary masks representing the read ROIs.
        The shape is (# ROIs, height, width).

    """
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
