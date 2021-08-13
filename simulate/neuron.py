import numpy as np
import random
import math
from skimage import draw
from skimage.morphology import binary_erosion, disk
from scipy.ndimage import gaussian_filter
from elasticdeform import deform_grid


NEURON_SIZE_MEAN = 10
NEURON_SIZE_VARIANCE = 10
MEMBRANE_THICKNESS_MIN = 3
MEMBRANE_THICKNESS_MAX = 5

BASELINE_BRIGHTNESS_MIN = 0.10 # 0.02
BASELINE_BRIGHTNESS_MAX = 0.25
CYTOSOL_DEPRESSION_MIN = 0.1
CYTOSOL_DEPRESSION_MAX = 1.0
SIGNAL_BOOST_FACTOR_MIN = 1.05
SIGNAL_BOOST_FACTOR_MAX = 1.35

NEURON_OPACITY = 0.5

NON_SPIKING_CELL_RATIO = 0.3
MAX_NUM_SPIKES_PER_FRAME = 0.02


class neuron:

    def __init__(self, ID, image_shape, roi_shape, deform=True):
        self.ID = ID
        self.image_shape = image_shape
        self._set_shape(roi_shape, deform)
        self._set_image()


    def _set_shape(self, roi_shape, deform):
        # Randomly choose the center coordinates within the region of interest
        y = random.randint(0, roi_shape[0]) + (self.image_shape[0] - roi_shape[0]) // 2
        x = random.randint(0, roi_shape[1]) + (self.image_shape[1] - roi_shape[1]) // 2

        # Model neuron shape as an ellipse
        angle = random.uniform(-math.pi, +math.pi)
        var_half = NEURON_SIZE_VARIANCE / 2
        axis1 = NEURON_SIZE_MEAN + random.randint(-var_half, +var_half)
        axis2 = NEURON_SIZE_MEAN + random.randint(-var_half, +var_half)
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


    def set_spikes(self, time_frames):
        num_spikes = random.randint(0, time_frames * MAX_NUM_SPIKES_PER_FRAME)
        spiking_frames = np.random.randint(0, time_frames, num_spikes)
        self.spiking_frames = np.sort(np.unique(spiking_frames))
        levels = np.random.uniform(SIGNAL_BOOST_FACTOR_MIN,
                                   SIGNAL_BOOST_FACTOR_MAX,
                                   len(self.spiking_frames))
        self.spike_levels = np.ones(time_frames)
        self.spike_levels[self.spiking_frames] = levels


    def add_cell_image(self, image, t):
        background = (1 - self.alpha) * image
        foreground = self.alpha * self.image * self.spike_levels[t]
        return foreground + background


    def add_temporal_mask(self, image, t):
        if(t in self.spiking_frames):
            return np.logical_or(image, self.mask)
        else:
            return image


    def add_spatial_mask(self, image):
        return np.logical_or(image, self.mask)
    