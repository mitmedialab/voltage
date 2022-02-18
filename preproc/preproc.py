import tifffile as tiff
from libpreproc import preprocess_video_cython
from skimage.transform import downscale_local_mean, resize


def preprocess_video(in_file, temporal_file, spatial_file,
                     signal_method, signal_period, signal_scale,
                     binning=1,
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
    signal_method : string
        Method for extracting signal from the corrected video. The options are
        max-median ('max-med'), median-min ('med-min'), and PCA ('pca').
    signal_period : integer
        Signal extraction will be performed per this time period (in frames).
    signal_scale : float
        Signal extraction will be performed after smoothing the corrected
        video with a spatial Gaussian filter of this scale (standard
        deviation).
    binning : integer, optional
        Downscale input video by spatial binning, where binning x binning
        pixels are averaged to yield one pixel. The default is 1 (no binning).
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

    in_image = tiff.imread(in_file).astype('float32')
    if(binning > 1):
        orig_shape = in_image.shape[1:]
        in_image = downscale_local_mean(in_image, (1, binning, binning))

    t, s = preprocess_video_cython(in_image,
                                   method_id, signal_period, signal_scale,
                                   num_threads)

    if(binning > 1):
        t = resize(t, (len(t),) + orig_shape, mode='constant')
        s = resize(s, (len(s),) + orig_shape, mode='constant')

    tiff.imwrite(temporal_file, t, photometric='minisblack')
    tiff.imwrite(spatial_file, s, photometric='minisblack')
