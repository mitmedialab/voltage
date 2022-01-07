import random
import math
import numpy as np
from scipy.ndimage import gaussian_filter


# shape parameters
STEP_SIZE = 1000 # for Bezier curve discretization
MIDPOINT_LOCATION_RATIO_MARGIN = 0.2
MIDPOINT_OFFSET_RATIO_MIN = 0.1
MIDPOINT_OFFSET_RATIO_MAX = 0.2

# flow parameters
BASELINE_INTENSITY_MIN = 0.01
BASELINE_INTENSITY_MAX = 0.05
FLOW_INTENSITY_MIN = 0.1
FLOW_INTENSITY_MAX = 0.3
FREQUENCY_MIN = 10 # in frames (one pulse per this many frames)
FREQUENCY_MAX = 20
WAVELENGTH_MIN = 10 # in pixels
WAVELENGTH_MAX = 30
WIDTH_MIN = 1.0 # in pixels (Gaussian sigma)
WIDTH_MAX = 3.0


class blood:
    
    def __init__(self, image_shape):
        """
        Initialize blood vessel instance.

        Parameters
        ----------
        image_shape : 2-tuple of integers
            Shape (H x W) of the image where the blood vessel will be placed.

        Returns
        -------
        None.

        """
        self.image_shape = image_shape
        self._set_shape(image_shape)
        self._set_flow()


    def _set_shape(self, image_shape):
        """
        Set the shape of the blood vessel. It is based on 3-point (quadratic)
        Bezier curve, where the both endpoints are randomly chosen within
        the image area and the midpoint is chosen such that the resultant
        curve is moderately curved (i.e., neither too straight nor too curved).

        Parameters
        ----------
        See __init__().

        Returns
        -------
        None.

        """
        # randomly pick a start point S
        self.sx = random.uniform(0, image_shape[1])
        self.sy = random.uniform(0, image_shape[0])
        
        # randomly pick an end point E
        self.ex = random.uniform(0, image_shape[1])
        self.ey = random.uniform(0, image_shape[0])

        # randomly pick a mid point M on the line S-E
        # (but not too close to the endpoints)
        ratio = random.uniform(MIDPOINT_LOCATION_RATIO_MARGIN,
                               1 - MIDPOINT_LOCATION_RATIO_MARGIN)
        self.mx = ratio * self.sx + (1 - ratio) * self.ex
        self.my = ratio * self.sy + (1 - ratio) * self.ey

        # offset M in the direction parpendicular to S-E
        # (offset amount is some fraction of the length of S-E)
        vx = self.ex - self.sx
        vy = self.ey - self.sy
        offset = random.uniform(MIDPOINT_OFFSET_RATIO_MIN,
                                MIDPOINT_OFFSET_RATIO_MAX)
        self.mx += offset * vy
        self.my -= offset * vx


    def _set_flow(self):
        """
        Set parameters for blood flow.

        Returns
        -------
        None.

        """        
        self.baseline_intensity = random.uniform(BASELINE_INTENSITY_MIN,
                                                 BASELINE_INTENSITY_MAX)
        self.flow_intensity = random.uniform(FLOW_INTENSITY_MIN,
                                             FLOW_INTENSITY_MAX)
        self.frequency = random.uniform(FREQUENCY_MIN, FREQUENCY_MAX)
        self.wavelength = random.uniform(WAVELENGTH_MIN, WAVELENGTH_MAX)
        self.width = random.uniform(WIDTH_MIN, WIDTH_MAX)
        
            
    def _draw_bezier(self, t):
        """
        Draw the Bezier curve representing the blood vessel. This function
        outputs one point on the curve given the curve parameter t, and hence
        it needs to be called multiple times to actually draw the curve.

        Parameters
        ----------
        t : float
            Curve parameter in [0, 1].

        Returns
        -------
        x : float
            X coordinate of the point on the curve corresponding to t.
        y : float
            Y coordinate of the point on the curve corresponding to t.
        mag : float
            Magnitude of the curve derivative at (x, y).

        """
        s = (1 - t) * (1 - t)
        m = 2 * (1 - t) * t
        e = t * t
        x = s * self.sx + m * self.mx + e * self.ex
        y = s * self.sy + m * self.my + e * self.ey
        
        # derivative
        ds = -2 + 2 * t # ds/dt
        dm = 2 - 4 * t # dm/dt
        de = 2 * t # de/dt
        dx = ds * self.sx + dm * self.mx + de * self.ex
        dy = ds * self.sy + dm * self.my + de * self.ey
        mag = math.sqrt(dx * dx + dy * dy)
        
        return x, y, mag

        
    def add_image(self, image, frame):
        """
        Add the blood vessel to a given image, taking into account the blood
        flow activity at a given frame.

        Parameters
        ----------
        image : 2D numpy.ndarray of float
            An image on which to draw the blood vessel.
        frame : integer
            A frame count.

        Returns
        -------
        2D numpy.ndarray of float
            Output image.

        """
        im = np.zeros(self.image_shape)
        length = 0
        for t in range(STEP_SIZE+1):
            x, y, mag = self._draw_bezier(t / STEP_SIZE)
            
            # The curve parameter t is incremented by dt = 1/STEP_SIZE.
            # The length of the curve segment corresponding to [t, t+dt]
            # is the derivative magnitude at this point multiplied by dt,
            # which is mag/STEP_SIZE. 
            mag /= STEP_SIZE
            
            # The curve length from the endpoint S to the current point (x, y)
            # can be obtained by integrating this curve segment length.
            length += mag
            
            # Intensity variation due to blood flow is modeled by a sinusoid.
            # The spatial interval between the peaks is 'wavelength' pixels.
            # The temporal interval between the peaks is 'frequency' frames,
            # so the blood pulse occurs once per 'frequency' frames. 
            flow = math.sin(2 * math.pi * (length / self.wavelength
                                           + frame / self.frequency))
            flow = 0.5 * (1 + flow) # [-1, +1] to [0, 1]

            # The intensity needs to be proportional to the curve segment
            # length because it is not constant for different t.
            mag *= self.baseline_intensity + self.flow_intensity * flow
            
            # Distribute the intensity to the four pixels around (x, y).
            ix = math.floor(x)
            iy = math.floor(y)
            fx = x - ix
            fy = y - iy
            if(ix < 0 or self.image_shape[1] <= ix+1):
                continue
            if(iy < 0 or self.image_shape[0] <= iy+1):
                continue
            im[iy,   ix]   += mag * (1 - fy) * (1 - fx)
            im[iy,   ix+1] += mag * (1 - fy) * fx
            im[iy+1, ix]   += mag * fy * (1 - fx)
            im[iy+1, ix+1] += mag * fy * fx
            
        # Turn the thin sharp curve into a thick smooth one by blurring
        # while compensating for the intensity reduction
        im = gaussian_filter(im, self.width) * self.width
        return image + im
