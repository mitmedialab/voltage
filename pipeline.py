import time
tic = time.time()
import multiprocessing
import tifffile as tiff
import sys
import argparse
import ntpath
import json
import os
import datetime
import cv2
import numpy as np
import gc
from multiprocessing import Pool
from cell_demix import demix_neurons
import ray
import nvgpu
from prettytable import PrettyTable

WEIGHT_PATH = '/home/ramdas/Documents/Voltage_Imaging/BestLoss_single_gpu.hdf5'
x = PrettyTable()
x.title = 'File run information'
x.field_names = ['Time', 'File tag', 'File read', 'Preprocess', 'U-Net', 'Demix', 'Total']


def parallel_resize(x, data, resize_val):
    return cv2.resize(data[x], resize_val)

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def printd(*args, **kwargs):
    s = " ".join(str(item) for item in args)
    print('[' + str(datetime.datetime.now())[:-3] + '] ' + s)

def segment_data(data, rd):
    import tensorflow as tf
    from model import initialize_unet
    import os
    os.environ['LD_LIBRARY_PATH']='/usr/local/cuda/lib64'
    session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    tf.Session(config = session_config)
    print("\n\n\n\nData:", data.shape)
    model = initialize_unet()
    model.load_weights(WEIGHT_PATH)
    rd['ypred'] = model.predict(data)


@ray.remote(num_gpus=2)
def preprocess_image(file, param, outpath_mc):
    from preprocess import preprocess
    mc, pre = preprocess(file, param)
    # tiff.imsave(outpath_mc, mc)
    return [mc, pre]

def get_minium_size_for_gpu():
    gpu_info = nvgpu.gpu_info()
    sizes = []
    SAFE_MEM = 1500
    for gi in gpu_info:
        sizes.append((gi['mem_total'] - gi['mem_used']))
    return min(sizes) - SAFE_MEM

def size_required(file):
    T = file.shape[0]
    H = max(128, file.shape[1])
    W = max(128, file.shape[2])
    return int((T * H * W * 4 * 2)/1e6)

def chunkIt(seqlen, num):
    avg = seqlen / float(num)
    out = []
    last = 0.0
    si = []
    ei = []
    while last < seqlen:
        si.append(int(last))
        ei.append(int(last + avg))
        last += avg
    return si, ei

def process_file(param, gsize):
    tic_full = time.time()
    magnification = param['magnification']
    param['output_dimension'] = 128
    fpath = param['filename']
    fname = path_leaf(fpath)

    outdir_mc = param['output_path'] + '/motion_corrected/'
    outpath_mc = outdir_mc + fname
    outdir_pre = param['output_path'] + '/preprocessed/'
    outpath_pre = outdir_pre + fname
    outdir_seg = param['output_path'] + '/segmented/'
    outpath_seg = outdir_seg + fname
    outdir_dmix = param['output_path'] + '/demixed/'
    outpath_dmix_json = outdir_dmix + fname[:-4] + '.json'
    outpath_dmix_mask = outdir_dmix + fname

    tic = time.time()
    file = tiff.imread(fpath).astype('float32')
    time_fileread = time.time() - tic
    T = file.shape[0]
    H = file.shape[1]
    W = file.shape[2]
    
    # Preprocessing
    time_saving = 0
    time_preprocess = 0
    if(os.path.exists(outpath_mc)):
        printd("[Filename: %s] - Motion corrected file exists! Skipping" %fname)
        tic = time.time()
        motion_corrected = tiff.imread(outpath_mc)
        time_preprocess = time.time() - tic
    else:   
        
        tic = time.time()
        if not os.path.exists(outdir_mc):
            os.mkdir(outdir_mc)

        size_req = size_required(file)
        if(size_req > gsize):
            si, ei = chunkIt(T, (size_req//gsize) + 1)
            mc = []
            pd = []
            for i in range(len(si)):
                s = si[i]
                e = ei[i]
                param['motion_correction']['length'] = (e-s)
                ret = preprocess_image.remote(np.array(file[s:e]), param, outpath_mc)
                ret = ray.get(ret)
                mc.append(ret[0])
                pd.append(ret[1])
            motion_corrected = np.concatenate(mc, axis=0)
            preprocessed = np.concatenate(pd, axis=0)
        else:
            param['motion_correction']['length'] = len(file)
            ret = preprocess_image.remote(file, param, outpath_mc)
            ret = ray.get(ret)

            motion_corrected = ret[0]
            preprocessed = ret[1]
        time_preprocess = time.time() - tic
        tic = time.time()
        tiff.imsave(outpath_mc, motion_corrected)
        time_saving += (time.time() - tic)


    # Preprocessing
    if(os.path.exists(outpath_pre)):
        tic = time.time()
        preprocessed = tiff.imread(outpath_pre)
        time_preprocess += time.time() - tic
    else:
        if not os.path.exists(outdir_pre):
            os.mkdir(outdir_pre)
        tic = time.time()
        tiff.imsave(outpath_pre, preprocessed)
        time_saving += (time.time() - tic)     


    if(os.path.exists(outpath_seg)):
        tic = time.time()
        ypred = tiff.imread(outpath_seg)
        time_pred = time.time() - tic
        time_postproc = 0
    else:
        tic = time.time()
        
        print("Prediction")
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        p = multiprocessing.Process(target=segment_data, args=(preprocessed,return_dict))
        p.start()
        p.join()
        ypred = return_dict['ypred']
        p.terminate()
        del manager, return_dict, p
        gc.collect()
        time_pred = time.time() - tic
        data=ypred
        resize_val = (W, H)

        tic = time.time()
        pool_out = []
        for i in range(data.shape[0]):
            pool_out.append(parallel_resize(i, data, resize_val))
        ypred = np.array(pool_out, 'float32')
        time_postproc = time.time() - tic
        if not os.path.exists(outdir_seg):
            os.mkdir(outdir_seg)
        tic = time.time()
        tiff.imsave(outpath_seg, ypred)
        time_saving += (time.time() - tic)      

    # Cell Demixing
    tic = time.time()
    if(not os.path.exists(outpath_dmix_json)):
        if not os.path.exists(outdir_dmix):
            os.mkdir(outdir_dmix)
        demix_data = demix_neurons(motion_corrected, ypred, param['expected_neurons'])
        with open(outpath_dmix_json, "w") as write_file:
            json.dump(demix_data['info'], write_file)
        tiff.imsave(outpath_dmix_mask, demix_data['mask']) 
    time_demix = time.time() - tic
    printd("Time fileread:", time_fileread)
    printd("Time preprocess:", time_preprocess)
    printd("Time unet:", time_pred)
    printd("Time post proc:", time_postproc)
    printd("Time demix:", time_demix)
    printd("Time saving:", time_saving)
    time_full = time.time() - tic_full
    x.add_row([str(datetime.datetime.now()), fname, str(round(time_fileread, 2)), 
        str(round(time_preprocess, 2)), str(round(time_pred + time_postproc, 2)), str(round(time_demix, 2)), str(round(time_full, 2))])



model = None
ray.init()
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--file',action='store',type=str)
group.add_argument('--batch',action='store_true')
args = parser.parse_args()

mode = 0
gsize = get_minium_size_for_gpu()


with open('file_params.txt') as file:
    params = json.load(file)

if(args.file == None):
    mode = 0
else:
    mode = 1
time_init = time.time() - tic
tic = time.time()
if(mode == 1):
    process_file(params[args.file], gsize)
else:
    for tag in params.keys():
        process_file(params[tag], gsize)
time_total = time.time() - tic

printd("\n\nTime init:", time_init)
printd("Total time:", round(time_total, 2), "seconds")
print(x)