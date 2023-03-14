import numpy as np
import tifffile as tiff
from libpreproc import preprocess_video_cython
from skimage.transform import resize_local_mean


def preprocess_video(in_file, in_video, temporal_file, spatial_file,
                     signal_method, signal_period, signal_scale,
                     downsampling=1.0,
                     num_threads=0):
    """
    Preprocess video for signal extraction.

    Parameters
    ----------
    in_file : string or pathlib.Path
        Input file path of a multi-page tiff containig motion/shading-corrected
        voltage imaging video. If this is empty, in_video will be used.
    in_video : 3D numpy.ndarray of float32
        Motion/shading-corrected voltage imaging video.
    temporal_file : string or pathlib.Path
        Output tiff file path to which extracted temporal signal will be saved.
    spatial_file : string or pathlib.Path
        Output tiff file path to which extracted spatial signal will be saved.
    signal_method : string
        Method for extracting signal from the corrected video. The options are
        max-median ('max-med'), median-min ('med-min'), and PCA ('pca').
    signal_period : integer
        Signal extraction will be performed per this time period (in frames).
    signal_scale : float
        Signal extraction will be performed after smoothing the corrected
        video with a spatial Gaussian filter of this scale (standard
        deviation).
    downsampling : float, optional
        Input video will be spatially downsampled by this factor. The width and
        height of the resultant video will be 1/downsampling of the original.
        The default is 1.0 (no downsampling).
    num_threads : integer, optional
        The number of threads to run the preprocessing on. The default is 0,
        in which case all the available cores will be used.

    Returns
    -------
    None.

    """
    if(signal_method == 'pca'):
        method_id = 0
    elif(signal_method == 'max-med'):
        method_id = 1
    elif(signal_method == 'med-min'):
        method_id = 2

    if(in_file):
        in_video = tiff.imread(in_file).astype('float32')

    t, s = preprocess_video_cython(in_video,
                                   method_id, signal_period, signal_scale,
                                   downsampling, num_threads)

    tiff.imwrite(temporal_file, t, photometric='minisblack')
    tiff.imwrite(spatial_file, s, photometric='minisblack')
