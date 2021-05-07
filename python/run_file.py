import time
tic_init = time.time()
import multiprocessing
import h5py
import tifffile as tiff
import sys
import ntpath
import json
import os
import threading
import queue
import datetime
import cv2
import numpy as np
import gc
from multiprocessing import Pool
from cell_demix import demix_neurons
import nvgpu
from preprocess import preprocess
from file_helper import fread, fwrite
from evaluate_each import get_f1_score
import pathlib
from postprocess import exp_spreading
save_queue = queue.Queue()

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def printd(*args, **kwargs):
    s = " ".join(str(item) for item in args)
    print('[' + str(datetime.datetime.now())[:-3] + '] ' + s)

def get_minium_size_for_gpu():
    gpu_info = nvgpu.gpu_info()
    sizes = []
    SAFE_MEM = 500
    for gi in gpu_info:
        sizes.append((gi['mem_total'] - gi['mem_used']))
    return min(sizes) - SAFE_MEM

# We would need twice the data size
def size_required_float(file):
    T = file.shape[0]
    H = max(128, file.shape[1])
    W = max(128, file.shape[2])
    return int((T * H * W * 4 * 2)/1e6)   

def parallel_save():
    while True:
        file = save_queue.get()
        if(file is None):
            return
        fwrite(file['name'], file['data'])

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

class pipeline:
    def __init__(self, tag, is_motion_correction_only=False, is_evaluate=False):

        self.gsize = get_minium_size_for_gpu()
        with open('settings.txt') as file:
            self.settings = json.load(file)

        with open('file_params.txt') as file:
            params = json.load(file)

        self.param = params[tag]
        self.exp_spread = bool(self.settings['exponential_spreading'])
        self.fpath = self.settings['input_base_path'] + '/' + self.param['filename']
        self.fname = self.param['filename']
        self.tag = tag
        self.outpath_base = self.settings['output_base_path'] + '/'

        self.mc_file = {}
        self.mc_file['path'] = self.outpath_base + self.settings['motion_correction_result_path'] + '/'
        self.mc_file['name'] = self.mc_file['path'] + self.fname

        self.pre_file = {}
        self.pre_file['path'] = self.outpath_base + self.settings['preprocessing_result_path'] + '/'
        self.pre_file['name'] = self.pre_file['path'] + self.fname

        self.seg_file = {}
        self.seg_file['path'] = self.outpath_base + self.settings['segmentation_result_path'] + '/'
        self.seg_file['name'] = self.seg_file['path'] + self.fname

        self.dmix_file = {}
        self.dmix_file['path'] = self.outpath_base + self.settings['cell_demixing_result_path'] + '/'
        self.dmix_file['json'] = self.dmix_file['path'] + pathlib.Path(self.fname).stem + '.json'
        self.dmix_file['mask'] = self.dmix_file['path'] + self.fname

        self.log_path = self.settings['output_base_path'] + '/' + self.settings['log_path'] + '/'
        self.execution_info_file = self.log_path + 'execution.info'

        self.eval_path = self.outpath_base + self.settings['evaluation_result_path'] + '/'
        self.eval_check_file = self.eval_path + self.fname

        self.weight_path = self.settings['weight_base_path'] + '/' + self.settings['weight_file']

        if not os.path.exists(self.mc_file['path']):
            os.mkdir(self.mc_file['path'])

        if not os.path.exists(self.pre_file['path']):
            os.mkdir(self.pre_file['path'])        

        if not os.path.exists(self.seg_file['path']):
            os.mkdir(self.seg_file['path'])

        if not os.path.exists(self.dmix_file['path']):
            os.mkdir(self.dmix_file['path'])

        if not os.path.exists(self.eval_path):
            os.mkdir(self.eval_path)

        self.time_saving = 0
        self.time_preprocess = 0
        self.time_pred = 0
        self.time_fileread = 0
        self.time_demix = 0
        self.time_eval = 0

        # Check segmentation first, so that we can start the parallel initialization
        if(is_motion_correction_only == False):
            if(os.path.exists(self.seg_file['name'])):
                self.run_segmentation = False
                tic = time.time()
                self.seg_file['data'] = fread(self.seg_file['name'])
                self.time_pred += time.time() - tic
            else:
                self.run_segmentation = True
                
                # Start the TF initialization and parallel model loading
                tic = time.time()
                self.manager = multiprocessing.Manager()
                self.return_dict = self.manager.dict()
                self.pred_q = multiprocessing.Queue()
                self.pred_proc = multiprocessing.Process(target=self.segment_data)
                self.pred_proc.start()
                self.time_pred += time.time() - tic
                self.data = None

            if(os.path.exists(self.dmix_file['json']) and os.path.exists(self.dmix_file['mask'])):
                self.run_demix = False
            else:
                self.run_demix = True

        else:
            self.run_demix = False
            self.run_segmentation = False

        self.run_eval = is_evaluate


        # Check existing preprocessed data
        if(os.path.exists(self.mc_file['name']) and os.path.exists(self.pre_file['name'])):
            self.run_preprocessing = False
            tic = time.time()
            self.mc_file['data'] = fread(self.mc_file['name'])
            self.pre_file['data'] = fread(self.pre_file['name'])
            self.T, self.H, self.W = self.mc_file['data'].shape
            self.time_preprocess += time.time() - tic
        else:
            self.run_preprocessing = True
            tic = time.time()
            self.file = fread(self.fpath).astype('float32')
            self.time_fileread += time.time() - tic
            self.T, self.H, self.W = self.file.shape


    def motion_correct(self, sq):
        tic = time.time()
        if(self.run_preprocessing):
            size_req = size_required_float(self.file)
            self.param['output_dimension'] = 128
            if(size_req > self.gsize):
                si, ei = chunkIt(self.T, (size_req//self.gsize) + 1)
                mc = []
                pd = []
                for i in range(len(si)):
                    s = si[i]
                    e = ei[i]
                    self.param['motion_correction']['length'] = (e-s)
                    try:
                        motion_corrected, preprocessed = preprocess(self.file[s:e], self.param)
                    except:
                        printd("[Error]: Motion Correction Caught an exception")
                        if(self.run_segmentation):
                            self.pred_q.put(None)
                        raise Exception("MotionCorrectionFailure")
                    mc.append(motion_corrected)
                    pd.append(preprocessed)
                
                self.mc_file['data'] = np.concatenate(mc, axis=0)
                self.pre_file['data'] = np.concatenate(pd, axis=0)
            else:
                self.param['motion_correction']['length'] = self.T
                try:
                    self.mc_file['data'], self.pre_file['data'] = preprocess(self.file, self.param)
                except:
                    printd("[Error]: Motion Correction Caught an exception")
                    if(self.run_segmentation):
                        self.pred_q.put(None)
                    raise Exception("MotionCorrectionFailure")
            self.time_preprocess += time.time() - tic
            tic = time.time()
            sq.put(self.mc_file)
            sq.put(self.pre_file)
            self.time_saving += time.time() - tic
        else:
            print("Finish mc")
            self.time_preprocess += time.time() - tic

    def segment_data(self):
        import tensorflow as tf
        from model import initialize_unet
        import os
        os.environ['LD_LIBRARY_PATH']='/usr/local/cuda/lib64'
        if(int(tf.__version__[0]) > 1):
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(e)
        else:
            session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
            tf.Session(config = session_config)
        try:
            model = initialize_unet()
            model.load_weights(self.weight_path)
        except:
            print("[Error]: Tensorflow caught an exception")
            return
        data = self.pred_q.get()
        if(data is None):
            return
        print("\nData:", data.shape)
        self.return_dict['ypred'] = model.predict(data)

    def prediction(self, sq):
        if(self.run_segmentation):
            printd("Starting prediction")
            tic = time.time()
            self.pred_q.put((self.pre_file['data'] * 255).astype('uint8'))
            self.pred_q.close()
            self.pred_q.join_thread()
            self.pred_proc.join()
            self.data = self.return_dict['ypred']
            self.pred_proc.terminate()
            print(self.data.shape)
            res = []
            for i in range(self.data.shape[0]):
                res.append(cv2.resize(self.data[i], (self.W, self.H)))
            ypred = np.array(res, 'float32')
            if(self.exp_spread == True):
                ypred = exp_spreading(ypred)
                ypred = ypred - ypred.min()
                ypred = ypred / ypred.max() 
            self.seg_file['data'] = ypred
            self.time_pred += (time.time() - tic)
            tic = time.time()
            sq.put(self.seg_file)
            self.time_saving += time.time() - tic

    def cell_demix(self):
        if(self.run_demix):
            tic = time.time()
            demix_data = demix_neurons(self.mc_file['data'], self.seg_file['data'], self.param['expected_neurons'])
            with open(self.dmix_file['json'], "w") as write_file:
                json.dump(demix_data['info'], write_file)
            fwrite(self.dmix_file['mask'], demix_data['mask']) 
            self.time_demix += time.time() - tic

    def dump_execution_info(self):
        execution_info = {}
        execution_info['time'] = str(datetime.datetime.now())
        execution_info['tag'] = self.tag
        execution_info['info'] = 'T: ' + str("%5d" %self.T) + ' | H: ' + str("%3d" %self.H) + ' | W: ' + str("%3d" %self.W)
        execution_info['file_read'] = str(round(self.time_fileread, 2))
        execution_info['preprocess'] = str(round(self.time_preprocess, 2))
        execution_info['prediction'] = str(round(self.time_pred, 2))
        execution_info['demix'] = str(round(self.time_demix, 2))
        if(self.run_eval):
            execution_info['eval'] = str(round(self.time_eval, 2))
            execution_info['f1'] = str(round(self.f1_at_T, 2))

        time_full = time.time() - tic_init
        execution_info['total'] = str(round(time_full, 2))
        with open(self.execution_info_file, 'a+') as f:
            f.write(json.dumps(execution_info) + '\n')

    def evaluate(self):
        if(self.run_eval):
            tic = time.time()
            self.eval_info = get_f1_score(self.settings, self.fname)
            self.f1_at_T = self.eval_info['consensus_f1'][np.where(self.eval_info['consensus_thresholds'] == 0.4)][0]
            self.time_eval += time.time() - tic


def main():

    tag = sys.argv[1]
    is_motion_correction_only = bool(int(sys.argv[2]))
    is_evaluate = bool(int(sys.argv[3]))

    print("\n\n")
    printd("Executing pipeline for file:", tag)
    print("\n")
    run_motion_correct = True
    run_prediction = True
    run_demix = True

    p = pipeline(tag, is_motion_correction_only, is_evaluate)

    
    t = threading.Thread(target=parallel_save, daemon=True)
    t.start()

    if(is_motion_correction_only):
        run_prediction = False
        run_demix = False

    if(run_motion_correct):
        p.motion_correct(save_queue)

    if(run_prediction):
        p.prediction(save_queue)

    if(run_demix):
        p.cell_demix()

    if(is_evaluate):
        p.evaluate()

    p.dump_execution_info()
    save_queue.put(None)
    t.join()

if __name__ == '__main__':
    main()





