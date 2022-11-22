import sys
import math
import numpy as np
from pathlib import Path
sys.path.append(str(Path('..').absolute()))
from evaluate import read_roi


INPUTS = [
    ('/media/bandy/nvme_data/voltage/datasets_v0.5/lowmag_GT', '*.tif'),
    ('/media/bandy/nvme_data/VolPy_Data/Extracted/voltage_L1', '*/*.zip'),
    ('/media/bandy/nvme_data/VolPy_Data/Extracted/voltage_TEG', '*/*.zip'),
    ('/media/bandy/nvme_data/VolPy_Data/Extracted/voltage_HPC', '*/*.zip'),
]


for input_dir, pat in INPUTS:
    print(input_dir)
    input_files = sorted(Path(input_dir).glob(pat))
    areas = []
    for in_file in input_files:
        masks = read_roi(in_file, (1000, 1000))
        areas.extend([np.count_nonzero(mask) for mask in masks])
    min_area = min(areas)
    max_area = max(areas)
    print('area range [%d, %d]  (sqrt [%.1f, %.1f])'
          % (min_area, max_area, math.sqrt(min_area), math.sqrt(max_area)))
