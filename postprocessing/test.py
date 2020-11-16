import numpy as np
import postprocess

file = np.load('file.npy')
frames = np.load('frames.npy')
sig = postprocess.postprocess(file[:5], frames)
