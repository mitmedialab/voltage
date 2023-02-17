import tifffile as tiff
import h5py
from libcorrect import correct_video_cython


def correct_video(in_file, motion_file, correction_file='',
                  first_frame=0, normalize=True,
                  motion_search_level=2, motion_search_size=5,
                  motion_patch_size=10, motion_patch_offset=7,
                  motion_x_range=1.0, motion_y_range=1.0,
                  shading_period=1000,
                  use_gpu=True, num_frames_per_batch=1000, num_threads=0):
    """
    Correct motion and shading in a video.

    Parameters
    ----------
    in_file : string or pathlib.Path
        Input file path of a multi-page tiff containig voltage imaging video.
    motion_file : string or pathlib.Path
        Output hdf5 file path to which estimated motion vectors will be saved.
    correction_file : string or pathlib.Path, optional
        Output tiff file path to which motion/shading-corrected video will be
        saved. The default is an empty string, in which case the video will be
        returned rather than saved.
    first_frame : integer, optional
        First frame number of the video. If nonzero, the input video will be
        treated as if it starts with that frame number by skipping the first
        frames. The default is zero.
    normalize : boolean, optional
        If True (default), the video intensity will be normalized.
    motion_search_level : integer, optional
        Max level of multiresolution motion correction. The default is 2.
        The GPU implementation does not support more than 2.
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
    motion_x_range : float, optional
        Fractional image range in X to be used in motion correction.
        For instance, 0.5 means patch-based correlation will be computed for
        50% of the image width around the center. The image outside of that
        range will not be referenced. This is useful when the image periphery
        does not contain much usable information to match.
        The default is 1.0 (full width).
    motion_y_range : float, optional
        Fractional image range in Y to be used in motion correction.
        The default is 1.0 (full height).
    shading_period : integer, optional
        Time period (in frames) for modeling shading variation.
        The default is 1000.
    use_gpu : boolean, optional
        GPUs will be used for the part of the algorithms whose GPU
        implementation is available. The default is True.
    num_frames_per_batch : integer, optional
        If use_gpu=True, the input video will be split into batches each having
        this number of frames, and will be processed one batch at a time.
        This will reduce the GPU memory usage and can improve performance as
        GPU data transfer can be overlapped with GPU processing.
        The default is 1000. This parameter will be ignored if use_gpu=False.
    num_threads : integer, optional
        The number of threads to run the preprocessing on. The default is 0,
        in which case all the available cores will be used.

    Returns
    -------
    c : 3D numpy.ndarray of float32
        Motion/shading-corrected video.

    """
    in_video = tiff.imread(in_file).astype('float32')
    in_video = in_video[first_frame:] # skip first frames
    c, x, y = correct_video_cython(in_video,
                                   1 if normalize else 0,
                                   motion_search_level, motion_search_size,
                                   motion_patch_size, motion_patch_offset,
                                   motion_x_range, motion_y_range,
                                   shading_period,
                                   1 if use_gpu else 0,
                                   num_frames_per_batch, num_threads)

    with h5py.File(motion_file, 'w') as f:
        f.create_dataset('x', data=x)
        f.create_dataset('y', data=y)

    if(correction_file):
        tiff.imwrite(correction_file, c, photometric='minisblack')
        return None
    else:
        return c
