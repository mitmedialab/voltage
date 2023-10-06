import os
import re
import runpy
import shutil
import pandas as pd
from pathlib import Path
from tifffile import TiffFile


# Path parameters
paths_file = Path(__file__).absolute().parents[2].joinpath('params', 'paths.py')
paths = runpy.run_path(paths_file)

INPUT_PATH = paths['HPC2_DATASETS']
OUTPUT_PATH = Path(paths['OUTPUT_BASE_PATH'], 'compare', 'volpy', 'voltage_HPC2')
FILENAME = ''               # If blank, all the files under INPUT_PATH will be processed
                            # Otherwise the specified file will be processed 10 times for runtime stats

# Computational resource parameters
NUM_PROCESSES = 4           # Number of processes to be used for motion correction and
                            # summary image creation (it seems that all the available
                            # threads will be used anyway though)
USE_CUDA = False            # Whether to use GPU for motion correction

# Motion correction parameters
DO_MOTION_CORRECTION = True # If False, previously saved result (mmap) will be used
MAX_SHIFT = 20              # Search range for motion correction (20 is equivalent to our 3)
SAVE_MOVIE = True           # Set False to measure motion correction computation time

# Summary image creation parameters
DO_SUMMARY_CREATION = True  # If False, previously saved result (tiff) will be used
GAUSSIAN_BLUR = False       # Set True when the input video is noisy

# Segmentation parameters
MIN_SIZE = 10               # Minimum neuron size to be detected
MAX_SIZE = 28               # Maximum neuron size to be detected
WEIGHTS_PATH = ''           # If blank, the default weights will be downloaded and used


def main(input_files):
    Path(OUTPUT_PATH).mkdir(exist_ok=True)
    for input_file in input_files:
        dataset_name = input_file.stem
        print('Processing ', dataset_name)

        # get exposure time from metadata to set frame rate
        exposure = None
        with TiffFile(input_file) as tif:
            for tag in tif.pages[0].tags.values():
                if(tag.name == 'ImageDescription'):
                    m = re.search('Exposure1\s*=\s*([\d\.]+)\s*s', tag.value)
                    exposure = float(m.group(1)) if m is not None else None
            if(tif.imagej_metadata is not None):
                for key, value in tif.imagej_metadata.items():
                    if(key == 'Info'):
                        m = re.search('Exposure1\s*=\s*([\d\.]+)\s*s', value)
                        exposure = float(m.group(1)) if m is not None else None
        if(exposure == None):
            print('exposure time was not found in metadata')
            exit()

        frame_rate = int(1 / exposure)
        output_dir = Path(OUTPUT_PATH).joinpath(dataset_name)
        args = (input_file, output_dir,
                NUM_PROCESSES, USE_CUDA,
                DO_MOTION_CORRECTION, MAX_SHIFT, SAVE_MOVIE,
                DO_SUMMARY_CREATION, frame_rate, GAUSSIAN_BLUR,
                MIN_SIZE, MAX_SIZE, WEIGHTS_PATH)
        command = 'python main.py %s %s %d %d %d %d %d %d %d %d %d %d %s' % args
        print(command)
        os.system(command)
        shutil.copy('time_motion_correct.txt', output_dir)
        shutil.copy('time_summary_images.txt', output_dir)
        shutil.copy('time_segmentation.txt', output_dir)


if(FILENAME): # run for a single file multiple times to get runtime statistics
    input_file = Path(INPUT_PATH, FILENAME)
    SAVE_MOVIE = True
    main([input_file]) # warm up run
    SAVE_MOVIE = False
    columns = ['Motion correction', 'Summary creation', 'Segmentation']
    df = pd.DataFrame(columns=columns)
    for i in range(10): # run 10 times
        main([input_file])
        times = []
        with open('time_motion_correct.txt') as f:
            times.append(float(f.readline()))
        with open('time_summary_images.txt') as f:
            times.append(float(f.readline()))
        with open('time_segmentation.txt') as f:
            times.append(float(f.readline()))
        df.loc[len(df)] = times
    m = df.mean()
    s = df.std()
    df.loc['Mean'] = m
    df.loc['Stdev'] = s
    df.to_csv(Path(OUTPUT_PATH, 'runtime.csv'))

else: # run for all the files under INPUT_PATH
    input_files = sorted(Path(INPUT_PATH).glob('*.tif'))
    main(input_files)
