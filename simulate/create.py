import numpy as np
import tifffile as tiff
from scipy.interpolate import interp1d, interp2d
from scipy.signal import windows
from scipy.ndimage import gaussian_filter
from skimage.util import random_noise

from .neuron import neuron
from .motion import synthesize_motion



# Todo: these should vary to some extent
LASER_SPOT_SIGMA = 50
SHOT_NOISE_SCALE = 1.0e4
SENSOR_READ_NOISE = 0.005
BACKGROUND_INTENSITY = 0.1

MISFOCUS_RATE = 0.01  # how often it happens
MISFOCUS_MAX = 50.0   # max spatial gaussian filter sigma
MISFOCUS_SPREAD = 5.0 # how long it lasts (temporal gaussian filter sigma)


def synthesize_background_fluorescence(image_shape):
    bg = np.ones(image_shape)
    ratio = image_shape[1] / image_shape[0]
    for i in range(image_shape[0] // 10, image_shape[0] // 4):
        j = i * ratio
        x = np.arange(0, i)
        y = np.arange(0, j)
        z = np.random.rand(len(x), len(y))
        f = interp2d(x, y, z)
        newx = np.arange(0, i-1, (i-1)/image_shape[0])
        newy = np.arange(0, j-1, (j-1)/image_shape[1])
        bg += f(newx[0:image_shape[0]], newy[0:image_shape[1]]) / i
    return bg


def synthesize_illumination_fluctuation(time_frames):
    temporal_profile = np.ones(time_frames)
    for i in range(time_frames // 100, time_frames):
        x = np.arange(0, i)
        y = np.random.rand(len(x))
        f = interp1d(x, y)
        newx = np.arange(0, i-1, (i-1)/time_frames)
        temporal_profile = temporal_profile + f(newx[0:time_frames]) / i
    temporal_profile /= np.mean(temporal_profile)
    return temporal_profile


def synthesize_focus_flucturation(time_frames):
    misfocus_frames = np.random.random(time_frames) < MISFOCUS_RATE
    num_misfocus = np.count_nonzero(misfocus_frames)
    offset = np.random.uniform(0, MISFOCUS_MAX, num_misfocus)
    focus_offset = np.zeros(time_frames)
    focus_offset[misfocus_frames] = offset
    return gaussian_filter(focus_offset, MISFOCUS_SPREAD)


def add_motion(image, out_shape, x, y):
    yofs = (image.shape[0] - out_shape[0]) // 2
    ys = int(yofs + y)
    ye = ys + out_shape[0]
    xofs = (image.shape[1] - out_shape[1]) // 2
    xs = int(xofs + x)
    xe = xs + out_shape[1]
    return image[ys:ye, xs:xe] # no subpixel shift


def add_illumination(image, sigma, scale):
    window_h = windows.gaussian(image.shape[0], sigma)
    window_w = windows.gaussian(image.shape[1], sigma)
    return image * window_h[:, np.newaxis] * window_w * scale


def add_occasional_misfocus(image, sigma):
    return gaussian_filter(image, sigma)


def add_shot_noise(image, scale):
    photon_counts = image * scale
    noisy = np.random.poisson(photon_counts)
    return noisy / scale


def add_read_noise(image, stdev):
    return random_noise(image, var=stdev**2)


def create_synthetic_data(image_shape, time_frames, num_neurons,
                          data_dir, temporal_gt_dir, spatial_gt_dir,
                          data_name):
    """
    Create synthetic voltage imaging data simulating a time-varying
    2D microscopy image, and save it as a multipage tiff file.

    Parameters
    ----------
    image_shape : 2-tuple of int
        Image height and width.
    time_frames : int
        Number of frames to synthesize.
    num_neurons : int
        Number of neurons in the image.
    data_dir : string
        Directory path where the simulated data will be saved.
    temporal_gt_dir : string
        Directory path where the temporal ground truth labeling will be saved,
        where each image represents the areas of active neurons at one frame.
    spatial_gt_dir : string
        Directory path where the spatial ground truth labeling will be saved,
        where each image represents the footprint of one neuron.
    data_name : string
        File name (used for both simulated data and the labelings).

    Returns
    -------
    None.

    """

    # set a canvas to draw neurons and other sythetic components on
    # to be larger than the output image size, because simulated motion
    # will cover a larger area
    canvas_shape = (image_shape[0] * 3 // 2, image_shape[1] * 3 // 2)

    # set an ROI within which neuron centers to be generated
    # to be a little smaller then the output image size,
    # to avoid neurons occupying a very small area in the image
    roi_shape = (image_shape[0] * 4 // 5, image_shape[1] * 4 // 5)
    
    # create neurons
    neurons = []
    for i in range(num_neurons):
        neu = neuron(i, canvas_shape, roi_shape)
        neu.set_spikes(time_frames)
        neurons.append(neu)

    # generate various synthetic components
    bg = BACKGROUND_INTENSITY * synthesize_background_fluorescence(canvas_shape)
    temporal_profile = synthesize_illumination_fluctuation(time_frames)
    focus_offset = synthesize_focus_flucturation(time_frames)
    Xs, Ys = synthesize_motion(time_frames)
    
    # synthesize video
    video = np.zeros((time_frames,) + image_shape)
    for t in range(time_frames):
        canvas = np.zeros(canvas_shape)
        for neu in neurons:
            canvas = neu.add_cell_image(canvas, t)
        canvas += bg
        frame = add_motion(canvas, image_shape, Xs[t], Ys[t])
        frame = add_illumination(frame, LASER_SPOT_SIGMA, temporal_profile[t])
        frame = add_occasional_misfocus(frame, focus_offset[t])
        frame = add_shot_noise(frame, SHOT_NOISE_SCALE)
        frame = add_read_noise(frame, SENSOR_READ_NOISE)
        video[t] = frame

    tiff.imwrite(data_dir + data_name + '.tif',
                 video.astype('float32'), photometric='minisblack')


    # generate corresponding ground truth labeling
    temporal_gt = np.zeros(video.shape, dtype=bool)
    for t in range(time_frames):
        canvas = np.zeros(canvas_shape, dtype=bool)
        for neu in neurons:
            canvas = neu.add_mask_image(canvas, t)
        temporal_gt[t] = add_motion(canvas, image_shape, 0, 0)

    tiff.imwrite(temporal_gt_dir + data_name + '.tif',
                 temporal_gt, photometric='minisblack')

    spatial_gt = np.zeros((0,) + image_shape, dtype=bool)
    for neu in neurons:
        if(neu.active): # skip non-spiking neurons
            canvas = np.zeros(canvas_shape, dtype=bool)
            canvas = neu.add_mask_image(canvas)
            crop = add_motion(canvas, image_shape, 0, 0)
            spatial_gt = np.append(spatial_gt, crop[np.newaxis], axis=0)
    
    tiff.imwrite(spatial_gt_dir + data_name + '.tif',
                 spatial_gt, photometric='minisblack')
    