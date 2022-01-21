import tifffile as tiff
import h5py
from libpreproc import preprocess_cython

def run_preprocessing(in_file, correction_file, motion_file,
                      temporal_file, spatial_file,
                      motion_search_level=2, motion_search_size=5,
                      motion_patch_size=10, motion_patch_offset=7,
                      shading_period=1000,
                      signal_method='max-med', signal_period=100,
                      signal_scale=0,
                      num_threads=0):
    """
    Preprocess images, which includes motion correction, shading correction,
    and signal extraction.

    Parameters
    ----------
    in_file : string
        Input file path of a multi-page tiff containig time-varying voltage
        imaging images.
    correction_file : string
        Output tiff file path to which motion/shading-corrected video will be
        saved.
    motion_file : string
        Output hdf5 file path to which estimated motion vectors will be saved.
    temporal_file : string
        Output tiff file path to which extracted temporal signal will be saved.
    spatial_file : string
        Output tiff file path to which extracted spatial signal will be saved.
    motion_search_level : int, optional
        Max level of multiresolution motion correction. The default is 2.
    motion_search_size : int, optional
        [-search_size, +search_size] pixels will be searched in X and Y at
        each level as motion vector candidates. The default is 5.
    motion_patch_size : int, optional
        [-patch_size, +patch_size] pixels will be used as a patch in X and Y
        when computing patch-based correlation in motion correction.
        The default is 10.
    motion_patch_offset : int, optional
        Offset (both in X and Y) between adjacent patches. A larger value
        leads to fewer patches with smaller overlaps. The default is 7.
    shading_period : int, optional
        Time period (in frames) for modeling shading variation.
        The default is 1000.
    signal_method : string, optional
        Method for extracting signal from the corrected video. The options are
        max-median ('max-med', default) and PCA ('pca').
    signal_period : int, optional
        Signal extraction will be performed per this time period (in frames).
        The default is 100.
    signal_scale : float, optional
        Signal extraction will be performed after smoothing the corrected
        video with a spatial Gaussian filter of this scale (standard
        deviation). The default is 0 (no smoothing).
    num_threads : int, optional
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
    c, t, s, x, y = preprocess_cython(in_image,
                                      motion_search_level, motion_search_size,
                                      motion_patch_size, motion_patch_offset,
                                      shading_period,
                                      signal_method_id, signal_period,
                                      signal_scale,
                                      num_threads)

    tiff.imwrite(correction_file, c, photometric='minisblack')
    tiff.imwrite(temporal_file, t, photometric='minisblack')
    tiff.imwrite(spatial_file, s, photometric='minisblack')
    with h5py.File(motion_file, 'w') as f:
        f.create_dataset('x', data=x)
        f.create_dataset('y', data=y)
