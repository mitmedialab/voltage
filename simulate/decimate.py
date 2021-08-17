import numpy as np
import tifffile as tiff


def decimate_video(in_file, out_file, size, mode):
    """
    Decimate video (time-varying 2D image) along time.

    Parameters
    ----------
    in_file : string
        File name of the input video.
    out_file : string
        File name of the output video.
    size : integer
        Window size for decimation. The number of frames in the output video
        will be the number of frames in the input video divided by this size.
    mode : string
        Decimation mode. The options are 'logical_or', 'mean', and 'median'.
        
    Returns
    -------
    None.

    """
    in_video = tiff.imread(in_file)
    num_in_frames = len(in_video)
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
