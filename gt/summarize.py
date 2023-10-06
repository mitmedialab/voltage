import runpy
from pathlib import Path
import numpy as np
import tifffile as tiff
import preproc


def _save_tiff(out_file, data):
    tiff.imwrite(out_file, data.astype('float32'), photometric='minisblack')


def summarize_video_for_manual_annotation(in_file, out_base):
    """
    Generate summary images to aid manual annotation of firing neurons.
    These include:
    - Average intensity projection along time,
    - Maximum intensity projection along time,
    - Maximum-minus-median projection for every 50-frame time segment, and
    - Patch-based first principal component for every 100-frame time segment.

    The input video must be motion corrected. If the data sets are not motion
    corrected, use pipeline.py with RUN_CORRECT=True to correct them first.

    Parameters
    ----------
    in_file : string of pathlib.Path
        Path to the input TIFF file containing motion-corrected video.
    out_base : string
        Path to the output directory in which summary images will be saved.

    Returns
    -------
    None.

    """

    in_video = tiff.imread(in_file)
    print(in_video.shape)

    # average intensity projection
    _save_tiff(out_base + '_avg.tif', np.mean(in_video, axis=0))

    # maximum intensity projection
    _save_tiff(out_base + '_max.tif', np.amax(in_video, axis=0))

    # max-median, spatial smoothing (signal_scale) is 2.0
    # so as not to blur the images too much for manual annotation 
    temporal_file = out_base + '_maxmed.tif'
    spatial_file = 'tmp.tif' # we don't need this
    preproc.preprocess_video(in_file, None, temporal_file, spatial_file,
                             'max-med', 50, 2.0)

    # PCA (turns out useless in most cases as it tends to extract blood vessels)
    temporal_file = out_base + '_pca.tif'
    preproc.preprocess_video(in_file, None, temporal_file, spatial_file,
                             'pca', 100, 0.0)


# The following code batch-processes files produced by pipeline.py
paths_file = Path(__file__).absolute().parents[1].joinpath('params', 'paths.py')
paths = runpy.run_path(paths_file)

INPUT_DIR = Path(paths['OUTPUT_BASE_PATH'], 'results', 'voltage_HPC2')
INPUT_FILES = sorted(INPUT_DIR.glob('*/*_corrected.tif'))

for in_file in INPUT_FILES:
    out_base = in_file.parent.joinpath(in_file.stem.replace('_corrected', ''))
    summarize_video_for_manual_annotation(in_file, str(out_base))
