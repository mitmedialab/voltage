import os
import re
import shutil
from pathlib import Path
from tifffile import TiffFile


INPUT_PATH = Path('/media/bandy/nvme_data/voltage/datasets_v0.5/lowmag')
OUTPUT_PATH = Path('/media/bandy/nvme_work/voltage/compare/volpy/lowmag')
WEIGHTS_PATH = ''  # if blank, the default weights will be downloaded and used
MIN_SIZE = 10
MAX_SIZE = 28
DO_MOTION_CORRECTION = True # if False, previously saved result (mmap) will be used
DO_SUMMARY_CREATION = True  # if False, previously saved result (tiff) will be used


input_files = sorted(INPUT_PATH.glob('*.tif'))
OUTPUT_PATH.mkdir(exist_ok=True)
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
    output_dir = OUTPUT_PATH.joinpath(dataset_name)
    command = 'python main.py %s %s %d %d %d %d %d %s' % (input_file,
                                                          output_dir,
                                                          frame_rate,
                                                          MIN_SIZE, MAX_SIZE,
                                                          DO_MOTION_CORRECTION,
                                                          DO_SUMMARY_CREATION,
                                                          WEIGHTS_PATH)
    print(command)
    os.system(command)
    shutil.copy('time_motion_correct.txt', output_dir)
    shutil.copy('time_summary_images.txt', output_dir)
    shutil.copy('time_segmentation.txt', output_dir)
