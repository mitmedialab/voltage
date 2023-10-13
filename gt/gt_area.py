import sys
import math
import runpy
import numpy as np
from pathlib import Path
sys.path.append(str(Path('..').absolute()))
from evaluate import read_roi


paths_file = Path(__file__).absolute().parents[1].joinpath('params', 'paths.py')
paths = runpy.run_path(paths_file)

INPUTS = [
    (Path(paths['HPC2_DATASETS'],  'HPC2_GT'),     '*.tif'),
    (Path(paths['VOLPY_DATASETS'], 'voltage_L1'),  '*/*.zip'),
    (Path(paths['VOLPY_DATASETS'], 'voltage_TEG'), '*/*.zip'),
    (Path(paths['VOLPY_DATASETS'], 'voltage_HPC'), '*/*.zip'),
]


for input_dir, pat in INPUTS:
    print(input_dir)
    input_files = sorted(input_dir.glob(pat))
    areas = []
    for in_file in input_files:
        masks = read_roi(in_file, (1000, 1000))
        areas.extend([np.count_nonzero(mask) for mask in masks])
    min_area = min(areas)
    max_area = max(areas)
    print('area range [%d, %d]  (sqrt [%.1f, %.1f])'
          % (min_area, max_area, math.sqrt(min_area), math.sqrt(max_area)))
