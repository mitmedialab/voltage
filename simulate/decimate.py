import numpy as np
import tifffile as tiff


def decimate_video(in_file, out_file, mode, size=0):
    """
    Decimate video (time-varying 2D image) along time.

    Parameters
    ----------
    in_file : pathlib.Path
        File name of the input video.
    out_file : pathlib.Path
        File name of the output video.
    mode : string
        Decimation mode. The options are 'logical_or', 'mean', and 'median'.
    size : integer, optional
        Window size for decimation. The number of frames in the output video
        will be the number of frames in the input video divided by this size.
        If size is zero (default), the entire video will be decimated into
        a single frame.
        
    Returns
    -------
    None.

    """
    in_video = tiff.imread(in_file)
    num_in_frames = len(in_video)
    if(size <= 0):
        size = num_in_frames
    num_out_frames = (num_in_frames + size - 1) // size
    out_video = np.zeros((num_out_frames,) + in_video[0].shape,
                         dtype=in_video.dtype)
    for i in range(num_out_frames):
        s = size * i
        e = min(s + size, num_in_frames)
        if(mode == 'logical_or'):
            out_video[i] = np.logical_or.reduce(in_video[s:e])
        elif(mode == 'mean'):
            out_video[i] = np.mean(in_video[s:e], axis=0)
        elif(mode == 'median'):
            out_video[i] = np.median(in_video[s:e], axis=0)
        else:
            print('Unsupported mode: ' + mode)
            return

    tiff.imsave(out_file, out_video, photometric='minisblack')
