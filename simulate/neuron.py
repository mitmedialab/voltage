import numpy as np
import random
import math
from skimage import draw
from skimage.morphology import binary_erosion, disk
from scipy.ndimage import gaussian_filter
from scipy.ndimage.measurements import center_of_mass
from elasticdeform import deform_grid


NEURON_SIZE_MIN = 10
NEURON_SIZE_MAX = 30
NEURON_ECCENTRICITY = 0.3 # used differently from mathematical definition
MEMBRANE_THICKNESS_MIN = 3
MEMBRANE_THICKNESS_MAX = 5

BASELINE_BRIGHTNESS_MIN = 0.15
BASELINE_BRIGHTNESS_MAX = 0.25
CYTOSOL_DEPRESSION_MIN = 0.1
CYTOSOL_DEPRESSION_MAX = 1.0
SIGNAL_BOOST_FACTOR_MIN = 1.15
SIGNAL_BOOST_FACTOR_MAX = 1.35

NEURON_OPACITY = 0.5

NON_SPIKING_CELL_RATIO = 0.7
NUM_SPIKES_PER_FRAME_MIN = 0.005
NUM_SPIKES_PER_FRAME_MAX = 0.02

NUM_DENDRITES_MIN = 0
NUM_DENDRITES_MAX = 2
DENDRITE_LENGTH_RATIO_MIN = 1 # relative to the cell body size
DENDRITE_LENGTH_RATIO_MAX = 3
DENDRITE_WIDTH_RATIO_MIN = 0.05
DENDRITE_WIDTH_RATIO_MAX = 0.1
ANGLE_OFFSET = 0.5 # how much angle can change from start to end
ANGLE_PERTURBATION = 0.1 # angle is randomized up to this value at each point


class neuron:

    def __init__(self, ID, image_shape, roi_shape, deform=True):
        """
        Initialize neuron instance.

        Parameters
        ----------
        ID : integer
            ID of the neuron.
        image_shape : 2-tuple of integers
            Shape (H x W) of the image where the neuron will be placed.
        roi_shape : 2-tuple of integers
            Shape (H x W) of the region-of-interest (ROI)
        deform : boolean, optional
            Whether or not to deform the base elliptic neuron shape
            in order to make it more realistic. The default is True.

        Returns
        -------
        None.

        """
        self.ID = ID
        self.image_shape = image_shape
        self._set_shape(roi_shape, deform)
        self._set_image()
        self._set_dendrites()


    def _set_shape(self, roi_shape, deform):
        """
        Set the shape of the neuron. It is a rotated and deformed ellipse
        whose center is within the ROI.

        Parameters
        ----------
        See __init__().

        Returns
        -------
        None.

        """
        # Randomly choose the center coordinates within the region of interest
        y = random.randint(0, roi_shape[0]) + (self.image_shape[0] - roi_shape[0]) // 2
        x = random.randint(0, roi_shape[1]) + (self.image_shape[1] - roi_shape[1]) // 2

        # Model neuron shape as an ellipse
        angle = random.uniform(-math.pi, +math.pi)
        axis1 = random.uniform(NEURON_SIZE_MIN, NEURON_SIZE_MAX) / 2
        axis2 = axis1 * random.uniform(1 - NEURON_ECCENTRICITY,
                                       1 + NEURON_ECCENTRICITY)
        Ys, Xs = draw.ellipse(y, x, axis1, axis2, self.image_shape, angle)

        self.mask = np.zeros(self.image_shape, dtype=bool)
        if(deform):
            tmp = np.zeros(self.image_shape)
            tmp[Ys, Xs] = 1.0
            num_points = 3
            displacement = np.random.rand(tmp.ndim, num_points, num_points)
            tmp = deform_grid(tmp, displacement * (axis1 + axis2))
            self.mask[tmp > 0.5] = True
        else:
            self.mask[Ys, Xs] = True


    # ToDo: add texture like perlin noise
    def _set_image(self):
        """
        Set the image of the neuron by assigning intensity over its footprint.

        Returns
        -------
        None.

        """
        # Initalize image with baseline brightness to cope with
        # the subsequent gaussian filtering and alpha composition
        baseline_brightness = random.uniform(BASELINE_BRIGHTNESS_MIN,
                                             BASELINE_BRIGHTNESS_MAX)
        img = np.ones(self.image_shape) * baseline_brightness

        # Depress cytosol intensity relative to membrane
        cytosol_depression = random.uniform(CYTOSOL_DEPRESSION_MIN,
                                            CYTOSOL_DEPRESSION_MAX)
        thickness = random.randint(MEMBRANE_THICKNESS_MIN,
                                   MEMBRANE_THICKNESS_MAX)
        cytosol = binary_erosion(self.mask, disk(thickness))
        img[cytosol] *= cytosol_depression

        sigma = 1.0 # slightly soften the boundary to reduce unrealistic jaggy
        self.image = gaussian_filter(img, sigma)
        opacity = NEURON_OPACITY * self.mask.astype(float)
        self.alpha = gaussian_filter(opacity, sigma)


    def _set_dendrites(self):
        """
        Set dendrites by growing thin strands from the cell body.
        An alpha map (self.dendrites) will be created separately from that
        of the cell body (self.alpha). The intensity reuses self.image.

        Returns
        -------
        None.

        """
        # center of the cell body
        y0, x0 = center_of_mass(self.mask)
        # size (diameter) of the cell body, assuming it is a disk
        size = 2 * math.sqrt(np.sum(self.mask) / math.pi)

        self.dendrites = np.zeros(self.image_shape)
        num_dendrites = random.randint(NUM_DENDRITES_MIN, NUM_DENDRITES_MAX)
        for i in range(num_dendrites):
            length = random.uniform(DENDRITE_LENGTH_RATIO_MIN,
                                    DENDRITE_LENGTH_RATIO_MAX) * size
            width = random.uniform(DENDRITE_WIDTH_RATIO_MIN,
                                   DENDRITE_WIDTH_RATIO_MAX) * size

            # randomly pick a direction of the dendrite at its root
            angle = random.uniform(-math.pi, +math.pi)
            # the direction at its endpoint will be angle + angle_offset
            angle_offset = random.random() * ANGLE_OFFSET

            # grow a dendrite starting from the center of the cell body
            x = x0
            y = y0
            angle_delta = angle_offset / length
            t = 0
            img = np.zeros(self.image_shape)
            for t in range(math.ceil(length)):
                # update the angle of the dendrite and take a unit step
                angle += angle_delta + random.random() * ANGLE_PERTURBATION
                x += math.cos(angle)
                y += math.sin(angle)
                xi = math.floor(x)
                yi = math.floor(y)
                if(xi < 0 or self.image_shape[1] <= xi+1):
                    continue
                if(yi < 0 or self.image_shape[0] <= yi+1):
                    continue
                if(self.mask[yi, xi]): # wait until it leaves the cell body
                    continue
                # The opacity at the root is the same as the cell body
                # but it wears off toward the endpoint
                img[yi, xi] = NEURON_OPACITY * (1 - t / length)

            # Turn the thin sharp curve into a thick smooth one by blurring
            # while compensating for the intensity reduction (i.e., undoing
            # Gaussian normalization), and add to the alpha map
            norm = math.sqrt(2 * math.pi) * width
            self.dendrites += gaussian_filter(img, width) * norm


    def set_spikes(self, time_frames):
        """
        Set a spiking/firing pattern of the neuron.

        Parameters
        ----------
        time_frames : integer
            The number of time frames over which to generate a spike profile.

        Returns
        -------
        None.

        """
        # first neuron (ID = 0) is always active to avoid synthesizing
        # data with no active neuron
        if(self.ID > 0 and random.random() < NON_SPIKING_CELL_RATIO):
            self.active = False
            num_spikes = 0
        else:
            self.active = True
            num_spikes = random.randint(time_frames * NUM_SPIKES_PER_FRAME_MIN,
                                        time_frames * NUM_SPIKES_PER_FRAME_MAX)
            
        spiking_frames = np.random.randint(0, time_frames, num_spikes)
        self.spiking_frames = np.sort(np.unique(spiking_frames))
        levels = np.random.uniform(SIGNAL_BOOST_FACTOR_MIN,
                                   SIGNAL_BOOST_FACTOR_MAX,
                                   len(self.spiking_frames))
        self.spike_levels = np.ones(time_frames) # base level is one
        self.spike_levels[self.spiking_frames] = levels


    def add_cell_image(self, image, t):
        """
        Add the neuron (image of the cell body and dendrites) to a given image.

        Parameters
        ----------
        image : 2D numpy.ndarray of float
            An image on which to draw the neuron.
        t : integer
            Time frame number.

        Returns
        -------
        2D numpy.ndarray of float
            Output image.

        """
        alpha = self.alpha + self.dendrites
        background = (1 - alpha) * image
        foreground = alpha * self.image * self.spike_levels[t]
        return foreground + background


    def add_mask_image(self, image, t=-1):
        """
        Add the neuron mask to a given image. The mask represents the cell body
        footprint and does not include dendrites.

        Parameters
        ----------
        image : 2D numpy.ndarray of boolean
            An image on which to draw the mask.
        t : integer, optional
            Time frame number. If the neuron is spiking at time t, the mask
            will be added. Otherwise the input image will be returned as-is.
            The default is -1, in which case the mask will always be added.

        Returns
        -------
        2D numpy.ndarray of boolean
            Output image.

        """
        if(t < 0 or t in self.spiking_frames):
            return np.logical_or(image, self.mask)
        else:
            return image
