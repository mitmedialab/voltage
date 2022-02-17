import tifffile as tiff
import h5py
from libcorrect import correct_video_cython


def correct_video(in_file, correction_file, motion_file,
                  first_frame=0,
                  motion_search_level=2, motion_search_size=5,
                  motion_patch_size=10, motion_patch_offset=7,
                  shading_period=1000,
                  num_threads=0):
    """
    Correct motion and shading in a video.

    Parameters
    ----------
    in_file : string or pathlib.Path
        Input file path of a multi-page tiff containig voltage imaging video.
    correction_file : string or pathlib.Path
        Output tiff file path to which motion/shading-corrected video will be
        saved.
    motion_file : string or pathlib.Path
        Output hdf5 file path to which estimated motion vectors will be saved.
    first_frame : integer, optional
        First frame number of the video. If nonzero, the input video will be
        treated as if it starts with that frame number by skipping the first
        frames. The default is zero.
    motion_search_level : integer, optional
        Max level of multiresolution motion correction. The default is 2.
    motion_search_size : integer, optional
        [-search_size, +search_size] pixels will be searched in X and Y at
        each level as motion vector candidates. The default is 5.
    motion_patch_size : integer, optional
        [-patch_size, +patch_size] pixels will be used as a patch in X and Y
        when computing patch-based correlation in motion correction.
        The default is 10.
    motion_patch_offset : integer, optional
        Offset (both in X and Y) between adjacent patches. A larger value
        leads to fewer patches with smaller overlaps. The default is 7.
    shading_period : integer, optional
        Time period (in frames) for modeling shading variation.
        The default is 1000.
    num_threads : integer, optional
        The number of threads to run the preprocessing on. The default is 0,
        in which case all the available cores will be used.

    Returns
    -------
    None.

    """
    in_image = tiff.imread(in_file).astype('float32')
    in_image = in_image[first_frame:] # skip first frames
    c, x, y = correct_video_cython(in_image,
                                   motion_search_level, motion_search_size,
                                   motion_patch_size, motion_patch_offset,
                                   shading_period,
                                   num_threads)

    tiff.imwrite(correction_file, c, photometric='minisblack')
    with h5py.File(motion_file, 'w') as f:
        f.create_dataset('x', data=x)
        f.create_dataset('y', data=y)
