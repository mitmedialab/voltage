import pandas as pd 
import json
import os
import tifffile as tiff
import h5py
from file_helper import fread, fwrite
from evaluate_each import evaluate_file
from evaluate_all import evaluate_all
import numpy as np
import glob
import pathlib

def read_hdf5(fname):
    with h5py.File(fname, 'r') as f:
        for key in f.keys():
            item = f[key]
            if(isinstance(item, h5py.Dataset) == True):
                return item.value

def write_hdf5(fname, data):
    with h5py.File(fname, 'w') as f:
        dset = f.create_dataset("default", data=data)

def get_run_info():

    with open('settings.txt') as file:
        settings = json.load(file)

    with open('file_params.txt') as file:
        params = json.load(file)

    log_path = settings['output_base_path'] + '/' + settings['log_path'] + '/'
    mask_path = settings['output_base_path'] + '/' + settings['cell_demixing_result_path'] + '/'
    pred_path = settings['output_base_path'] + '/' + settings['segmentation_result_path'] + '/'

    exec_info_file = log_path + 'execution.info'

    info = {}
    info['tags'] = []
    info['is_finished'] = False
    info['tinfo'] = pd.DataFrame(columns=['S. No.', 'Time', 'File tag', 'File info', 'File read time', 'Preprocess time', 'U-Net time', 'Demix time', 'Total time'])
    if(os.path.exists(exec_info_file)):
        with open(exec_info_file, 'r') as f:
            content = f.readlines()
        content = [x.strip() for x in content] 
        
        
        
        total_count = 0
        for l in content:
            d = json.loads(l)
        
            if(d['tag'] == 'None'):
                if(d['msg'] == 'FinishedExecution'):
                    info['is_finished'] = True
                    continue

            if(d['tag'] not in info['tags']):
                info['tags'].append(d['tag'])
                info[d['tag']] = {}
                info[d['tag']]['fname'] = params[d['tag']]['filename']
                info[d['tag']]['count'] = 0
                info[d['tag']]['tinfo'] = pd.DataFrame(columns=['S. No.', 'Time', 'File tag', 'File info', 'File read time', 'Preprocess time', 'U-Net time', 'Demix time', 'Total time'])

            if('eval' in d.keys()):
                if(total_count == 0):
                    info['tinfo'] = pd.DataFrame(columns=['S. No.', 'Time', 'File tag', 'File info', 'File read time', 'Preprocess time', 'U-Net time', 'Demix time', 'Total time', 'Eval Score @ IoU=0.4'])
                info['tinfo'].loc[total_count] = [str('%3d' %(total_count+1)), d['time'], d['tag'], d['info'], d['file_read'], d['preprocess'], d['prediction'], d['demix'], d['total'], d['f1']]

                if(info[d['tag']]['count'] == 0):
                    info[d['tag']]['tinfo'] = pd.DataFrame(columns=['S. No.', 'Time', 'File tag', 'File info', 'File read time', 'Preprocess time', 'U-Net time', 'Demix time', 'Total time', 'Eval Score @ IoU=0.4'])
                info[d['tag']]['tinfo'].loc[info[d['tag']]['count']] = [str('%3d' %(total_count+1)), d['time'], d['tag'], d['info'], d['file_read'], d['preprocess'], d['prediction'], d['demix'], d['total'], d['f1']]
            else:
                info['tinfo'].loc[total_count] = [str('%3d' %(total_count+1)), d['time'], d['tag'], d['info'], d['file_read'], d['preprocess'], d['prediction'], d['demix'], d['total']]
                info[d['tag']]['tinfo'].loc[info[d['tag']]['count']] = [str('%3d' %(total_count+1)), d['time'], d['tag'], d['info'], d['file_read'], d['preprocess'], d['prediction'], d['demix'], d['total']]

            try:
                info[d['tag']]['mask'] = fread(mask_path + params[d['tag']]['filename'])
                info[d['tag']]['pred'] = np.average(fread(pred_path + params[d['tag']]['filename']), axis=0)
            except:
                pass
            info[d['tag']]['count'] += 1
            total_count += 1

            
    return info

def get_eval_info(fname):

    with open('settings.txt') as file:
        settings = json.load(file)

    return evaluate_file(settings, fname, save=False)

def get_eval_all_info():
    return evaluate_all(save=False)


def if_eval_ready(fname):
    with open('settings.txt') as file:
        settings = json.load(file)

    basename = pathlib.Path(fname).stem

    eval_path = settings['output_base_path'] + '/' + settings['evaluation_result_path'] + '/'
    if(len(glob.glob(eval_path + basename + '*')) == 8):
        return True
    else:
        return False