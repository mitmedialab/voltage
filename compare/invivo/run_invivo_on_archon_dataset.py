from pathlib import Path
from main import run_invivo_segmentation


INPUT_PATH = '/media/bandy/nvme_data/voltage/datasets_v0.5/lowmag'
OUTPUT_PATH = '/media/bandy/nvme_work/voltage/compare/invivo/lowmag'


max_block_size = 50
th = 4
cut_off_point = 0.95 # 0.8-0.99
length_cut = 10      # minimum pixel count of a superpixel
length_max = 1000    # maximum pixel count of a superpixel
patch_size = 30
corr_th_fix = 0.4    # 0.3-0.6

input_files = sorted(Path(INPUT_PATH).glob('*.tif'))
Path(OUTPUT_PATH).mkdir(exist_ok=True)
for input_file in input_files:
    dataset_name = input_file.stem
    print('Processing ', dataset_name)
    output_dir = Path(OUTPUT_PATH).joinpath(dataset_name)
    run_invivo_segmentation(input_file, output_dir,
                            max_block_size,
                            th, cut_off_point,
                            length_cut, length_max, patch_size,
                            corr_th_fix)
