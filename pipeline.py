import os
import ntpath
import glob
import multiprocessing as mp
import tifffile as tiff

from simulate import create_synthetic_data, decimate_temporal_gt
from demix import demix_cells
from evaluate import run_ipynb_evaluate_each, run_ipynb_evaluate_all


IMAGE_SHAPE = (128, 128)
TIME_FRAMES = 1000
TIME_SEGMENT_SIZE = 50
NUM_VIDEOS = 10
BASE_PATH = '/media/bandy/nvme_work/voltage/sim/test'


def set_dir(dirname):
    path = BASE_PATH + '/' + dirname
    if not os.path.exists(path):
        os.mkdir(path)
    return path + '/'
    

def simulate(num_videos, data_dir, temporal_gt_dir, spatial_gt_dir):
    args = []
    for i in range(num_videos):
        name = '%4.4d' % i
        num_neurons = i % 10 + 5
        args.append((IMAGE_SHAPE, TIME_FRAMES, num_neurons,
                     data_dir, temporal_gt_dir, spatial_gt_dir, name))
    
    pool = mp.Pool(mp.cpu_count())
    pool.starmap(create_synthetic_data, args)
    pool.close()


def decimate(in_dir, out_dir):
    filenames = glob.glob(in_dir + '/*.tif')
    filenames.sort()
    args = []
    for in_file in filenames:
        out_file = out_dir + ntpath.basename(in_file)
        args.append((in_file, out_file, TIME_SEGMENT_SIZE))
    
    pool = mp.Pool(mp.cpu_count())
    pool.starmap(decimate_temporal_gt, args)
    pool.close()


def preprocess(in_dir, out_dir, correction_dir):
    filenames = glob.glob(in_dir + '/*.tif')
    filenames.sort()
    for in_file in filenames:
        command = 'preproc/main -db -ms 5 -sm 0 -sc 0 -sw %d ' % TIME_SEGMENT_SIZE + in_file + ' ' + out_dir
        print(command)
        os.system(command)
        
        out_file = ntpath.basename(in_file)
        command = 'mv ' + out_dir + '/signal.tif ' + out_dir + out_file
        os.system(command)
        command = 'mv ' + out_dir + '/corrected.tif ' + correction_dir + out_file
        os.system(command)
        

def demix(in_dir, out_dir):
    filenames = glob.glob(in_dir + '/*.tif')
    filenames.sort()
    for in_file in filenames:
        probability_maps = tiff.imread(in_file)
        masks = demix_cells(probability_maps, threshold=0.1)
        out_file = out_dir + ntpath.basename(in_file)
        tiff.imwrite(out_file, masks.astype('float32'), photometric='minisblack')


def evaluate(in_dir, gt_dir, out_dir):
    filenames = glob.glob(in_dir + '*.tif')
    filenames.sort()
    for in_file in filenames:
        gt_file = gt_dir + ntpath.basename(in_file)
        run_ipynb_evaluate_each(in_file, gt_file, out_dir)
    run_ipynb_evaluate_all(out_dir)




mode = 'sim'

if(mode == 'toy'):
    do_simulate = True
    do_decimate = True
    do_preprocess = False
    do_demix = True
    do_evaluate = True
elif(mode == 'sim'):
    do_simulate = True
    do_decimate = False
    do_preprocess = True
    do_demix = True
    do_evaluate = True


if(do_simulate):
    data_dir = set_dir('data')
    temporal_gt_dir = set_dir('temporal_label')
    spatial_gt_dir = set_dir('spatial_label')
    simulate(NUM_VIDEOS, data_dir, temporal_gt_dir, spatial_gt_dir)

if(do_decimate):
    decimated_gt_dir = set_dir('temporal_label_%d' % TIME_SEGMENT_SIZE)
    decimate(temporal_gt_dir, decimated_gt_dir)

if(do_preprocess):
    preprocess_dir = set_dir('preprocessed')
    correction_dir = set_dir('corrected')
    preprocess(data_dir, preprocess_dir, correction_dir)

if(do_demix):
    demix_dir = set_dir('demixed')
    if(do_decimate):
        demix(decimated_gt_dir, demix_dir)
    else:
        demix(preprocess_dir, demix_dir)

if(do_evaluate):
    eval_dir = set_dir('evaluated')
    evaluate(demix_dir, spatial_gt_dir, eval_dir)
