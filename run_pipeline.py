import time
import sys
import argparse
import os
import datetime
import json
from prettytable import PrettyTable
import subprocess
import signal
import multiprocessing
from python.evaluate_all import prepare_evaluate_all_notebook

def printd(*args, **kwargs):
    s = " ".join(str(item) for item in args)
    print('[' + str(datetime.datetime.now())[:-3] + '] ' + s)

shutdown = False

def handle_signal(signum, frame):
    global shutdown
    printd("Caught signal... Stopping execution!")
    shutdown = True
    
signal.signal(signal.SIGQUIT, handle_signal)

class pipeline_info:
    def __init__(self, task='batch', 
                is_motion_correct_only=False, 
                is_evaluate=False, is_gui=False, 
                mode="temporal", num_attempts=3):


        if(task == 'batch'):
            self.is_batch = True
        else:
            self.is_batch = False
            self.file_tag = task

        with open('settings.txt') as file:
            self.settings = json.load(file)

        with open('file_params.txt') as file:
            self.params = json.load(file)

        if not os.path.exists(self.settings['output_base_path']):
            os.mkdir(self.settings['output_base_path'])

        self.log_path = self.settings['output_base_path'] + '/' + self.settings['log_path'] + '/'
        self.log_file = self.log_path + 'execution.log'
        self.execution_info_file = self.log_path + 'execution.info'

        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)

        if(os.path.exists(self.execution_info_file)):
            os.remove(self.execution_info_file)
        if(os.path.exists(self.log_file)):
            os.remove(self.log_file)

        self.motion_correct_only = is_motion_correct_only
        self.evaluate = is_evaluate
        self.evaluate_path = self.settings['output_base_path'] + '/' + self.settings['evaluation_result_path']
        self.num_attempts = num_attempts
        self.dash_port = self.settings['dash_port']
        self.gui = is_gui
        self.mode = mode

    def run_file(self, tag=None):
        global shutdown
        
        if(tag is None):
            tag = self.file_tag

        printd("Executing pipeline for file:", str(tag))

        if(self.mode == "temporal"):
            fmode = 0
        elif(self.mode == "spatial"):
            fmode = 1
        else:
            mag = self.params[tag]['magnification']
            if(mag == 16):
                fmode = 0
            else:
                fmode = 1
        cmd = 'python python/run_file.py ' + str(tag) + ' ' + str(int(self.motion_correct_only)) + ' ' + str(int(self.evaluate)) + ' ' + str(fmode)
        for retries in range(self.num_attempts):
            try:
                subprocess.run(cmd, 
                    stdout=open(self.log_file, 'a+'), 
                    stderr=open(self.log_file, 'a+'), 
                    shell=True, check=True)

            except Exception as e:
                printd("Caught exception from subprocess execution:", e)
                if(shutdown == True):
                    break
                if(retries != 2):
                    printd("Got error while running file: " + str(tag) + " . Retrying!")
                else:
                    printd("Got error while running file: " + str(tag) + " . No more retries!")
            else:
                break

    def run_batch(self):
        for tag in self.params.keys():
            self.run_file(tag)
            if(shutdown == True):
                break

    def print_pretty(self):
        x = PrettyTable()
        x.title = 'File run information'
        if(self.evaluate == True):
            x.field_names = ['S. No.', 'Time', 'File tag', 'File info', 'File read', 'Preprocess', 'U-Net', 'Demix', 'Total', 'F1@(T=0.4)']
        else:
            x.field_names = ['S. No.', 'Time', 'File tag', 'File info', 'File read', 'Preprocess', 'U-Net', 'Demix', 'Total']

        if(os.path.exists(self.execution_info_file)):
            with open(self.execution_info_file, 'r') as f:
                content = f.readlines()
            content = [x.strip() for x in content] 
            count = 0
            for l in content:
                count += 1
                d = json.loads(l)
                if(self.evaluate):
                    x.add_row([str('%3d' %count), d['time'], d['tag'], d['info'], d['file_read'], d['preprocess'], d['prediction'], d['demix'], d['total'], d['f1']])
                else:
                    x.add_row([str('%3d' %count), d['time'], d['tag'], d['info'], d['file_read'], d['preprocess'], d['prediction'], d['demix'], d['total']])
            print(x)
        else:
            print("No files executed in this run")

    def check_evaluate(self):
        if(self.evaluate == True and self.is_batch == True):
            eval_info = prepare_evaluate_all_notebook(self.evaluate_path)
            F1 = round(eval_info['rep'][0], 4)
            F116 = round(eval_info['rep'][1], 4)
            F120 = round(eval_info['rep'][2], 4)
            F140 = round(eval_info['rep'][3], 4)
            printd("Finished batch evaluation")
            printd("Overall F1 score (Aggregate of TP,FP,FN): %0.4f" %F1)
            printd("F1 score info (Aggregate of TP,FP,FN): 16x: %0.4f, 20x: %0.4f, 40x: %0.4f" %(F116, F120, F140))
            
            F1 = round(eval_info['each'][0]['F1'].mean(), 4)
            F116 = round(eval_info['each'][1]['F1'].mean(), 4)
            F120 = round(eval_info['each'][2]['F1'].mean(), 4)
            F140 = round(eval_info['each'][3]['F1'].mean(), 4)
            F1s = round(eval_info['each'][0]['F1'].std(), 4)
            F116s = round(eval_info['each'][1]['F1'].std(), 4)
            F120s = round(eval_info['each'][2]['F1'].std(), 4)
            F140s = round(eval_info['each'][3]['F1'].std(), 4)

            printd("Overall F1 score (Mean ± Std): %0.4f ± %0.4f" %(F1, F1s))
            printd("F1 score info (Mean ± Std): 16x: %0.4f ± %0.4f, 20x: %0.4f ± %0.4f, 40x: %0.4f ± %0.4f" %(F116, F116s, F120, F120s, F140, F140s))

    def signal_finished(self):
        execution = {}
        execution['tag'] = 'None'
        execution['msg'] = 'FinishedExecution'
        with open(self.execution_info_file, 'a+') as f:
            f.write(json.dumps(execution) + '\n')
        


def get_args(): 
    parser = argparse.ArgumentParser()
    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument('--file', action='store', type=str)
    group1.add_argument('--batch', action='store_true')
    group2 = parser.add_mutually_exclusive_group(required=False)
    group2.add_argument('--motion-correct-only', action='store_true')
    group2.add_argument('--evaluate',action='store_true')
    group3 = parser.add_mutually_exclusive_group(required=True)
    group3.add_argument('--temporal', action='store_true')
    group3.add_argument('--spatial', action='store_true')
    group3.add_argument('--mix', action='store_true')
    parser.add_argument('--gui', action='store_true', required=False)
    args = parser.parse_args()

    if(args.batch == True):
        task = 'batch'
    else:
        task = args.file

    mode = "temporal"
    if(args.spatial == True):
        mode = "spatial"
    elif(args.mix == True):
        mode = "mix"
    p = pipeline_info(task, args.motion_correct_only, args.evaluate, args.gui, mode)

    return p




def main():

    tic = time.time()

    # Get the pipeline execution paramters and 
    # perform necessary initlizations
    p = get_args()

    if(p.gui == True):
        gp = subprocess.Popen(['python', 'python/dash_interface.py', str(p.dash_port), str(int(p.evaluate))])

    if(p.evaluate == True):
        ep = subprocess.Popen(['python', 'python/evaluation_runner.py'])
    # Execute the project
    printd("Starting the execution")
    if(p.is_batch == True):
        p.run_batch()
    else:
        p.run_file()


    # Print the timing information
    p.print_pretty()
    time_total = time.time() - tic
    printd("Total time:", time_total)
    
    p.signal_finished()

    if(shutdown == False):
        if(p.evaluate == True):
            printd("Waiting for file evaluation completion...")
            ep.wait()
            printd("Finished file evaluations")

        # Check and run evaluation is required
        p.check_evaluate()

        if(p.gui == True):
            a = input("Press ENTER key to exit the GUI.")
            os.killpg(os.getpgid(gp.pid), signal.SIGTERM)

if __name__ == '__main__':
    main()
