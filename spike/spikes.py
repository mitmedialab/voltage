import numpy as np
import h5py
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


def detect_spikes(video, masks, out_file, polarity, spike_thresh):
    """
    Extract voltage traces from video and masks, detect spikes, and save them.

    Parameters
    ----------
    video : 3D numpy.ndarray of float
        Motion/shading-corrected video of neurons.
    masks : 3D numpy.ndarray of boolean
        ROI masks representing neuron footprints.
    out_file : string or pathlib.Path
        Path to a file to which spike information will be saved.
    polarity : integer
        1 for voltage indicators with positive polarity (fluorescence increases
        for higher voltage) and -1 for negative polarity (decreases)
    spike_thresh : float
        The number of standard deviations above (below in the case of negative
        polarity) which voltage is considered spiking

    Returns
    -------
    None.

    """
    with h5py.File(out_file, 'w') as f:
        for i, mask in enumerate(masks):
            v = np.mean(video, axis=(1, 2), where=mask)
            v -= gaussian_filter1d(v, 10)
            v -= np.mean(v)
            thresh = spike_thresh * np.std(v)
            if(polarity < 0):
                v = -v
            peaks, _ = find_peaks(v, height=thresh)
            
            grp = f.create_group('neuron%d' % i)
            grp.create_dataset('voltage', data=v)
            grp.create_dataset('spikes', data=peaks)
            grp.attrs['thresh']=thresh
