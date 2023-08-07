# Default parameters for pipeline.py
# Overwrite them using a separate parameter file

#%% Common parameters (must be consistent between training and inference)

TIME_SEGMENT_SIZE = 50  # Input video will be split into time segments
                        # having this number of frames, and segmentation of
                        # active neurons will be done for each time segment


#%% Motion/shading correction parameters

FIRST_FRAME = 0            # Start processing input video from this frame
                           # as some datasets have corrupted beginning

NORMALIZE = True           # Whether to perform intensity normalization

MOTION_SEARCH_LEVEL = 2    # Level of multiresolution motion correction
                           # Level 2 means input video will be downsampled
                           # two times (in half each time)

MOTION_SEARCH_SIZE = 3     # How far motion vector will be searched in X and Y
                           # One level of multiresolution effectively doubles
                           # this size (i.e., level 2 quadruples it)

MOTION_PATCH_SIZE = 10     # Patch size for template matching (e.g., 10 means
                           # 21 x 21 pixels

MOTION_PATCH_OFFSET = 7    # Offset/interval/stride between sliding patches

MOTION_X_RANGE = 1.0       # Range in X of image region where sliding patches
                           # cover (e.g., 0.5 means half of the image width)
                           # Useful when image periphery is dark/featureless
MOTION_Y_RANGE = 1.0       # Range in Y

SHADING_PERIOD = 1000      # Shading correction will be performed for each
                           # time period of this length in frames


#%% Preprocessing parameters

SIGNAL_DOWNSAMPLING = 1.0  # Video frames will be downsampled by this factor
                           # (2.0 means half) before signal extraction

SIGNAL_SCALE = 3.0         # Scale (standard deviation) of spatial Gaussian
                           # filter to be applied to each frame of video
                           # before signal extraction

SIGNAL_METHOD = 'max-med'  # Signal extraction method
                           # 'max-med': maximum minus median
                           # 'med-min': median minus minimum (for Voltron)


#%% Segmentation parameters

TILE_SHAPE = (64, 64)        # Shape (height, width) of tiles to be extracted
                             # from preprocessed video: segmentation will be
                             # performed for each tile and the results will be
                             # merged to produce active neuron probability maps
                             
TILE_STRIDES = (8, 8)        # Offset/interval/stride between sliding tiles

TILE_MARGIN = (0, 0)         # Margin that tiles will leave on the image border,
                             # specified as ratios to the image height and width

BATCH_SIZE = 128             # Batch size for U-Net inference

NORM_CHANNEL = 1             # Each tile will be normalized with respect to
                             # this channel (spatial signal in our case)

NORM_SHIFTS = [False, True]  # Whether to shift the values of each channel
                             # so their minimum within tile becomes zero


#%% Demixing parameters

PROBABILITY_THRESHOLD = 0.5  # Probability value in [0, 1] above which pixels
                             # are considered belonging to firing neurons
                             
AREA_THRESHOLD_MIN = 0       # Regions having no less area than this will be
                             # extracted as neurons
                             
AREA_THRESHOLD_MAX = 10000   # Regions having no greater area than this will be
                             # extracted as neurons
                             
CONCAVITY_THRESHOLD = 10     # Regions having no greater concavity than this
                             # will be extracted as neurons, where concavity is
                             # calculated as the area of the convex hull of the
                             # region divided by the area of the region

INTENSITY_THRESHOLD = 0      # Regions having no less mean intensity than this
                             # will be extracted as neurons

ACTIVITY_THRESHOLD = 0       # Regions having no less activity level than this
                             # will be extracted as neurons, where activity
                             # level is calculated as the product of the mean
                             # intensity and the mean firing probability

BACKGROUND_SIGMA = 10        # Gaussian filter size (standard deviation) to
                             # estimate a background intensity map

BACKGROUND_EDGE = 1.0        # When the background is estimated with Gaussian
                             # filtering, pixels near the image borders tend to
                             # be dark and can produce false foreground there
                             # This parameter can be used to mitigate this
                             # and specifies how far from the image borders
                             # this treatment should reach

BACKGROUND_THRESHOLD = 0.003 # Image regions having larger intensity than the
                             # background by this value will be extracted as
                             # foreground

MASK_DILATION = 0            # The computed masks will be dilated by this size
                             # This may be useful for accuracy evaluation
                             # against manual annotation because humans tend to
                             # to draw slightly larger ROIs around neurons


#%% Spike detection parameters

POLARITY        = 1          # 1 for voltage indicators with positive polarity
                             # (fluorescence increases for higher voltage) and
                             # -1 for negative polarity (fluorescence decreases
                             # for higher voltage like Voltron)

SPIKE_THRESHOLD = 2.5        # Neurons are considered spiking when their
                             # voltage is larger than its subthreshold activity
                             # range by this number of times

REMOVE_INACTIVE = False      # Whether to remove non-spiking neurons


#%% Evaluation parameters

REPRESENTATIVE_IOU = 0.4     # IoU threshold at which representative F-1 scores
                             # will be computed


#%% Runtime parameters

RUN_MODE = 'run'     # Mode for running the pipeline
                     # 'run'   : run the pipeline for neuron detection
                     # 'train' : train U-Net segmentation network

RUN_SIMULATE = True  # Whether to simulate video to synthesize training data
RUN_CORRECT  = True  # Whether to run motion/shading correction
RUN_PREPROC  = True  # Whether to run preprocessing
RUN_SEGMENT  = True  # Whether to run segmentation
RUN_DEMIX    = True  # Whether to run demixing
RUN_SPIKE    = True  # Whether to run spike detection
RUN_EVALUATE = True  # Whether to evaluate segmentation accuracy and speed
RUN_TRAIN    = True  # Whether to train U-Net


#%% Performance parameters (optimal values depend on the computer environment)

#NUM_GPUS = 2              # If unspecified, all available GPUs will be used
                           # for motion correction and segmentation

USE_GPU_CORRECT = True     # Use GPU (if available) for motion correction

BATCH_SIZE_CORRECT = 1000  # Number of frames to be processed in one batch
                           # for GPU motion correction: a smaller size can
                           # better overlap computation and data transfer,
                           # but too small a value may incur overhead

NUM_THREADS_CORRECT = 0    # Number of threads for motion/shading correction
                           # 0 uses all the available logical CPU cores

NUM_THREADS_PREPROC = 0    # Number of threads for preprocessing
                           # 0 uses all the available logical CPU cores

GPU_MEM_SIZE = 5           # GPU memory size in gigabytes used for segmentation
                           # Keep it well below the max GPU memory capacity to
                           # avoid GPU out of memory error
