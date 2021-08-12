import numpy as np
import tifffile as tiff


def decimate_temporal_gt(in_file, out_file, size):
    in_video = tiff.imread(in_file).astype(bool)
    num_in_frames = len(in_video)
    num_out_frames = num_in_frames // size
    out_video = np.zeros((num_out_frames,) + in_video[0].shape, dtype=bool)
    for i in range(num_out_frames):
        s = size * i
        e = s + size
        out_video[i] = np.logical_or.reduce(in_video[s:e])

    tiff.imsave(out_file, out_video, photometric='minisblack')
