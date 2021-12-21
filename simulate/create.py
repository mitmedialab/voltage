import numpy as np
import random
import tifffile as tiff
from scipy.interpolate import interp1d, interp2d
from scipy.signal import windows
from scipy.ndimage import gaussian_filter
from skimage.util import random_noise

from .neuron import neuron
from .motion import synthesize_motion
from .blood import blood


MAX_NUM_OVERLAPS = 2

LASER_SPOT_SIGMA_MIN = 40
LASER_SPOT_SIGMA_MAX = 50
SHOT_NOISE_SCALE_MIN = 1.0e4
SHOT_NOISE_SCALE_MAX = 2.0e4
SENSOR_READ_NOISE_MIN = 0.003
SENSOR_READ_NOISE_MAX = 0.007
BACKGROUND_INTENSITY_MIN = 0.1
BACKGROUND_INTENSITY_MAX = 0.2

MISFOCUS_RATE = 0.01  # how often it happens
MISFOCUS_MAX = 50.0   # max spatial gaussian filter sigma
MISFOCUS_SPREAD = 5.0 # how long it lasts (temporal gaussian filter sigma)

NUM_BLOOD_VESSELS = 3


def synthesize_background_fluorescence(image_shape):
    """
    Synthesize a Perlin-noise-like background image.

    Parameters
    ----------
    image_shape : 2-tuple of int
        Height and width of the image to be synthesized.

    Returns
    -------
    bg : 2D numpy.ndarray of float
        Synthesized background image.

    """
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
    """
    Synthesize a Perlin-noise-like 1D array modeling fluctuating illumination.

    Parameters
    ----------
    time_frames : int
        The number of time frames (the length of the array) to be synthesized.

    Returns
    -------
    temporal_profile : 1D numpy.ndarray of float
        A sequence of values modeling temporal illumination fluctuation.

    """
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
    """
    Synthesize a 1D array modeling lens focus fluctuation (occasional defocus).
    Each value in the array represents a degree of defocus (0 means in-focus),
    and occasional defocus is modeled as random numbers in [0, MISFOCUS_MAX)
    at some frames ocurring at a probability of MISFOCUS_RATE. To make the
    transitions smooth from in-focus to defocus and back to in-focus, the array
    will be Gaussian-filtered with a standard deviation of MISFOCUS_SPREAD.

    Parameters
    ----------
    time_frames : int
        The number of time frames (the length of the array) to be synthesized.

    Returns
    -------
    1D numpy.ndarray of float
        A sequence of values modeling focus fluctuation.

    """
    misfocus_frames = np.random.random(time_frames) < MISFOCUS_RATE
    # let first 10 frames stay in focus for motion correction
    misfocus_frames[0:min(10, time_frames)] = False
    num_misfocus = np.count_nonzero(misfocus_frames)
    offset = np.random.uniform(0, MISFOCUS_MAX, num_misfocus)
    focus_offset = np.zeros(time_frames)
    focus_offset[misfocus_frames] = offset
    return gaussian_filter(focus_offset, MISFOCUS_SPREAD)


def add_motion(image, out_shape, x, y):
    """
    Add motion to an image by shifting it by specified offsets, and crop it
    so that the output image will have a specified shape. The reason for
    cropping is because we model a larger image than we want to synthesize
    (image.shape should be larger than out_shape) so that the output image
    will not contain undefined (outside of the input image) values.

    Parameters
    ----------
    image : 2D numpy.ndarray of float
        Input image.
    out_shape : 2-tuple of int
        Height and width of the output image.
    x : float
        Offset (motion vector) in X.
    y : float
        Offset (motion vector) in Y.

    Returns
    -------
    2D numpy.ndarray of float
        Shifted and cropped image.

    """
    yofs = (image.shape[0] - out_shape[0]) // 2
    ys = int(yofs + y)
    ye = ys + out_shape[0]
    xofs = (image.shape[1] - out_shape[1]) // 2
    xs = int(xofs + x)
    xe = xs + out_shape[1]
    return image[ys:ye, xs:xe] # no subpixel shift


def add_illumination(image, sigma, scale):
    """
    Add illumination effects to an image using a 2D Gaussian, modeling
    a bright center and dark periphery that a laser spot would produce.

    Parameters
    ----------
    image : 2D numpy.ndarray of float
        Input image.
    sigma : float
        Standard deviation of the 2D Gaussian modeling the laser spot width.
    scale : float
        Illumination intensity.

    Returns
    -------
    2D numpy.ndarray of float
        Output image.

    """
    window_h = windows.gaussian(image.shape[0], sigma)
    window_w = windows.gaussian(image.shape[1], sigma)
    return image * window_h[:, np.newaxis] * window_w * scale


def add_occasional_misfocus(image, sigma):
    """
    Add occasional misfocus effects to an image. Misfocus is modeled by
    a 2D Gaussian filter.

    Parameters
    ----------
    image : 2D numpy.ndarray of float
        Input image.
    sigma : float
        Standard deviation of the 2D Gaussian modeling misfocus.

    Returns
    -------
    2D numpy.ndarray of float
        Output image.

    """
    return gaussian_filter(image, sigma)


def add_shot_noise(image, scale):
    """
    Add shot noise to an image, which is modeled as Poisson noise.

    Parameters
    ----------
    image : 2D numpy.ndarray of float
        Input image.
    scale : float
        Scaling constant converting a pixel value into the number of photons.

    Returns
    -------
    2D numpy.ndarray of float
        Output image.

    """
    photon_counts = image * scale
    noisy = np.random.poisson(photon_counts)
    return noisy / scale


def add_read_noise(image, stdev):
    """
    Add sensor read noise to an image, which is modeled as Gaussian noise.

    Parameters
    ----------
    image : 2D numpy.ndarray of float
        Input image.
    stdev : float
        Standard deviation of the Gaussian noise.

    Returns
    -------
    2D numpy.ndarray of float
        Output image.

    """
    return random_noise(image, var=stdev**2)


def create_synthetic_data(image_shape, time_frames, time_segment_size,
                          num_neurons,
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
    time_segment_size : int
        Number of frames in the time segment used for preprocessing.
        If a positive value is specified, different background noise components
        are generated for every time segment in order to augment training data.
        Preprocessing should use the same time segment size.
        If zero is specified, noise components are generated only once at the
        beginning and will be consistent throughout the data.
    num_neurons : int
        Number of neurons in the image.
    data_dir : pathlib.Path
        Directory path where the simulated data will be saved.
    temporal_gt_dir : pathlib.Path
        Directory path where the temporal ground truth labeling will be saved,
        where each image represents the areas of active neurons at one frame.
    spatial_gt_dir : pathlib.Path
        Directory path where the spatial ground truth labeling will be saved,
        where each image represents the footprint of one neuron.
    data_name : string
        File name (used for both simulated data and the labelings).

    Returns
    -------
    None.

    """

    print('synthesizing ' + data_name)

    # set a canvas to draw neurons and other sythetic components on
    # to be larger than the output image size, because simulated motion
    # will cover a larger area
    canvas_shape = (image_shape[0] * 3 // 2, image_shape[1] * 3 // 2)

    # set an ROI within which neuron centers to be generated
    # to be a little smaller then the output image size,
    # to avoid neurons occupying a very small area in the image
    roi_shape = (image_shape[0] * 4 // 5, image_shape[1] * 4 // 5)
    
    # create neurons while avoiding too many overlaps
    count = np.zeros(canvas_shape, dtype=int)
    neurons = []
    for i in range(num_neurons):
        while(True):
            neu = neuron(i, canvas_shape, roi_shape)
            if(np.amax(count + neu.mask) <= MAX_NUM_OVERLAPS):
                count += neu.mask
                break
        neu.set_spikes(time_frames)
        neurons.append(neu)

    # generate one-time synthetic components
    laser_spot_sigma = random.uniform(LASER_SPOT_SIGMA_MIN,
                                      LASER_SPOT_SIGMA_MAX)
    temporal_profile = synthesize_illumination_fluctuation(time_frames)
    focus_offset = synthesize_focus_flucturation(time_frames)
    Xs, Ys = synthesize_motion(time_frames)
    bloods = []
    for i in range(NUM_BLOOD_VESSELS):
        bloods.append(blood(canvas_shape))
    
    # synthesize video
    video = np.zeros((time_frames,) + image_shape)
    for t in range(time_frames):
        # randomly change noisy background every time segment
        # unless time_segment_size is zero
        if((time_segment_size == 0 and t == 0)
           or (time_segment_size > 0 and t % time_segment_size == 0)):
            bg_val = random.uniform(BACKGROUND_INTENSITY_MIN,
                                    BACKGROUND_INTENSITY_MAX)
            bg = bg_val * synthesize_background_fluorescence(canvas_shape)
            shot_noise_scale = random.uniform(SHOT_NOISE_SCALE_MIN,
                                              SHOT_NOISE_SCALE_MAX)
            sensor_read_noise = random.uniform(SENSOR_READ_NOISE_MIN,
                                               SENSOR_READ_NOISE_MAX)

        canvas = np.zeros(canvas_shape)
        for neu in neurons:
            canvas = neu.add_cell_image(canvas, t)
        for bld in bloods:
            canvas = bld.add_image(canvas, t)
        canvas += bg
        frame = add_motion(canvas, image_shape, Xs[t], Ys[t])
        frame = add_illumination(frame, laser_spot_sigma, temporal_profile[t])
        frame = add_occasional_misfocus(frame, focus_offset[t])
        frame = add_shot_noise(frame, shot_noise_scale)
        frame = add_read_noise(frame, sensor_read_noise)
        video[t] = frame

    tiff.imwrite(data_dir.joinpath(data_name + '.tif'),
                 video.astype('float32'), photometric='minisblack')


    # generate corresponding ground truth labeling
    temporal_gt = np.zeros(video.shape, dtype=bool)
    for t in range(time_frames):
        canvas = np.zeros(canvas_shape, dtype=bool)
        for neu in neurons:
            canvas = neu.add_mask_image(canvas, t)
        temporal_gt[t] = add_motion(canvas, image_shape, 0, 0)

    tiff.imwrite(temporal_gt_dir.joinpath(data_name + '.tif'),
                 temporal_gt, photometric='minisblack')

    spatial_gt = np.zeros((0,) + image_shape, dtype=bool)
    for neu in neurons:
        if(neu.active): # skip non-spiking neurons
            canvas = np.zeros(canvas_shape, dtype=bool)
            canvas = neu.add_mask_image(canvas)
            crop = add_motion(canvas, image_shape, 0, 0)
            spatial_gt = np.append(spatial_gt, crop[np.newaxis], axis=0)
    
    tiff.imwrite(spatial_gt_dir.joinpath(data_name + '.tif'),
                 spatial_gt, photometric='minisblack')
    