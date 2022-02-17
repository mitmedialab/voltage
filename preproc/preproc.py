import tifffile as tiff
from libpreproc import preprocess_video_cython


def preprocess_video(in_file, temporal_file, spatial_file,
                     first_frame=0,
                     signal_method='max-med', signal_period=100,
                     signal_scale=0,
                     num_threads=0):
    """
    Preprocess video for signal extraction.

    Parameters
    ----------
    in_file : string or pathlib.Path
        Input file path of a multi-page tiff containig motion/shading corrected
        voltage imaging video.
    temporal_file : string or pathlib.Path
        Output tiff file path to which extracted temporal signal will be saved.
    spatial_file : string or pathlib.Path
        Output tiff file path to which extracted spatial signal will be saved.
    first_frame : integer, optional
        First frame number of the video. If nonzero, the input video will be
        treated as if it starts with that frame number by skipping the first
        frames. The default is zero.
    signal_method : string, optional
        Method for extracting signal from the corrected video. The options are
        max-median ('max-med', default) and PCA ('pca').
    signal_period : integer, optional
        Signal extraction will be performed per this time period (in frames).
        The default is 100.
    signal_scale : float, optional
        Signal extraction will be performed after smoothing the corrected
        video with a spatial Gaussian filter of this scale (standard
        deviation). The default is 0 (no smoothing).
    num_threads : integer, optional
        The number of threads to run the preprocessing on. The default is 0,
        in which case all the available cores will be used.

    Returns
    -------
    None.

    """
    if(signal_method == 'PCA'):
        signal_method_id = 0
    else: # max-median
        signal_method_id = 1

    in_image = tiff.imread(in_file).astype('float32')
    in_image = in_image[first_frame:] # skip first frames
    t, s = preprocess_video_cython(in_image,
                                   signal_method_id, signal_period,
                                   signal_scale,
                                   num_threads)

    tiff.imwrite(temporal_file, t, photometric='minisblack')
    tiff.imwrite(spatial_file, s, photometric='minisblack')
