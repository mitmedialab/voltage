import numpy as np
import tifffile as tiff
import h5py
import multiprocessing as mp
from threading import Thread
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


def _mean_roi(video, masks, idx, results):
    voltages = np.zeros((len(masks), len(video)))
    for i, mask in enumerate(masks):
        voltages[i] = np.mean(video, axis=(1, 2), where=mask)
    results[idx] = voltages

def extract_voltage_traces(video, masks, num_threads):
    """
    Extract voltage traces from video for given ROI masks.

    Parameters
    ----------
    See detect_spikes() for the definitions of other parameters.

    Returns
    -------
    2D numpy.ndarray of float
        Extracted voltage traces. The shape is (# masks, # video frames).

    """
    if(num_threads == 0):
        num_threads = mp.cpu_count()
    threads = [None] * num_threads
    results = [None] * num_threads
    num_frames = len(video)
    num_frames_per_thread = (num_frames + num_threads - 1) // num_threads
    for i in range(num_threads):
        s = num_frames_per_thread * i
        e = min(s + num_frames_per_thread, num_frames)
        threads[i] = Thread(target=_mean_roi,
                            args=(video[s:e], masks, i, results))
        threads[i].start()

    for t in threads:
        t.join()

    return np.concatenate(results, axis=1)


def detect_spikes_sub(voltage, polarity, spike_thresh):
    """
    Detect spikes of a voltage trace.

    Parameters
    ----------
    voltage: 1D numpy.ndarray of float
        Input raw voltage trace.

    See detect_spikes() for the definitions of other parameters.

    Returns
    -------
    voltage : 1D numpy.ndarray of float
        Rectified voltage trace.
    spikes : 1D numpy.ndarray of integer
        Spike times (indices to the voltage trace).
    scale : float
        Scale of the trace. It is the original dynamic (pixel intensity) range
        of the estimated subthreshold acitivity of the voltage trace.

    """
    if(polarity < 0):
        voltage = -voltage
    # Remove baseline fluctuation (filter size may need to be adjusted)
    voltage -= gaussian_filter1d(voltage, 100)
    # Zero-centering (median works better than mean because spikes drag
    # the mean toward positive side)
    voltage -= np.median(voltage)
    # Use 1 percentile to estimate the subthreshold activity range [p1, -p1]
    # (p1 is negative) as it seems to work better than standard deviation
    # which tends to overestimate the range for actively spiking neurons
    p1 = np.percentile(voltage, 1)
    scale = -p1
    # Normalize voltage trace by the subthreshold activity range
    voltage /= scale
    # Detect spikes when voltage is larger than the subthreshold activity range
    # by spike_thresh times
    spikes, _ = find_peaks(voltage, height=spike_thresh)
    return voltage, spikes, scale


def detect_spikes(video, masks, spike_file, mask_file, polarity, spike_thresh,
                  remove_inactive=False, num_threads=0):
    """
    Extract voltage traces from video and masks, detect spikes, and save them.
    Masks for inactive neurons (with no detected spikes) will be removed.

    Parameters
    ----------
    video : 3D numpy.ndarray of float
        Motion/shading-corrected video of neurons.
    masks : 3D numpy.ndarray of boolean
        ROI masks representing neuron footprints.
    spike_file : string or pathlib.Path
        Path to a file to which spike information will be saved.
    mask_file : string or pathlib.Path
        Path to a file to which refined ROI masks (where those for inactive
        neurons with no detected spikes have been removed) will be saved.
    polarity : integer
        1 for voltage indicators with positive polarity (fluorescence increases
        for higher voltage) and -1 for negative polarity (decreases).
    spike_thresh : float
        Neurons are considered spiking when their voltage is larger than its
        subthreshold activity range by spike_thresh times.
    remove_inactive : boolean
        Whether to remove inactive (non-spiking) neurons. Default is False.
    num_threads : integer, optional
        The number of threads to be used. Default is 0, in which case all the
        available cores will be used.

    Returns
    -------
    None.

    """
    voltage_list = extract_voltage_traces(video, masks, num_threads)

    inactive_neurons = []
    with h5py.File(spike_file, 'w') as f:
        neuron_id = 0
        for i, voltage in enumerate(voltage_list):
            v, s, t = detect_spikes_sub(voltage, polarity, spike_thresh)
            if(remove_inactive and len(s) == 0):
                inactive_neurons.append(i)
            else:
                grp = f.create_group('neuron%d' % neuron_id)
                grp.create_dataset('voltage', data=v)
                grp.create_dataset('spikes', data=s)
                grp.attrs['scale']=t
                neuron_id += 1

    masks = np.delete(masks, inactive_neurons, axis=0)
    if(len(masks) == 0):
        # add a blank image so the file has at least one page
        masks = np.zeros((1,) + masks.shape[1:], dtype=bool)
    tiff.imwrite(mask_file, masks.astype('uint8') * 255,
                 photometric='minisblack')
