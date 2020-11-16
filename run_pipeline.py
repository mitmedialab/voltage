import time
import sys
import argparse
import os
import datetime
import json
from prettytable import PrettyTable
import subprocess
import signal
from python.evaluate import evaluate_all

def printd(*args, **kwargs):
    s = " ".join(str(item) for item in args)
    print('[' + str(datetime.datetime.now())[:-3] + '] ' + s)

shutdown = False

def handle_signal(signum, frame):
    global shutdown
    printd("Caught signal... Stopping execution after current file!")
    shutdown = True
    
signal.signal(signal.SIGQUIT, handle_signal)

class pipeline_info:
    def __init__(self, task='batch', 
                is_new_project=False, 
                is_motion_correct_only=False, 
                is_evaluate=False, num_attempts=3):


        if(task == 'batch'):
            self.is_batch = True
        else:
            self.is_batch = False
            self.file_tag = task

        self.new_project = is_new_project

        with open('settings.txt') as file:
            self.settings = json.load(file)

        with open('file_params.txt') as file:
            self.params = json.load(file)


        self.log_path = self.settings['output_base_path'] + '/' + self.settings['log_path'] + '/'
        self.log_file = self.log_path + 'execution.log'
        self.timing_file = self.log_path + 'file_timing.info'

        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)

        if(is_new_project == True):
            if(os.path.exists(self.timing_file)):
                os.remove(self.timing_file)
            if(os.path.exists(self.log_file)):
                os.remove(self.log_file)

        self.motion_correct_only = is_motion_correct_only
        self.evaluate = is_evaluate
        self.evaluate_path = self.settings['output_base_path'] + '/' + self.settings['evaluation_result_path']
        self.num_attempts = num_attempts

    def run_file(self, tag=None):
        if(tag is None):
            tag = self.file_tag

        printd("Executing pipeline for file:", str(tag))
        cmd = 'python python/run_file.py ' + str(tag) + ' ' + str(int(self.motion_correct_only)) + ' ' + str(int(self.evaluate))
        for retries in range(self.num_attempts):
            try:
                subprocess.run(cmd, 
                    stdout=open(self.log_file, 'a+'), 
                    stderr=open(self.log_file, 'a+'), 
                    shell=True, check=True)

            except Exception as e:
                printd("Caught exception from subprocess execution:", e)
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
        x.title = 'File run timing information'
        if(self.evaluate == True):
            x.field_names = ['S. No.', 'Time', 'File tag', 'File info', 'File read', 'Preprocess', 'U-Net', 'Demix', 'Eval', 'Total']
        else:
            x.field_names = ['S. No.', 'Time', 'File tag', 'File info', 'File read', 'Preprocess', 'U-Net', 'Demix', 'Total']

        if(os.path.exists(self.timing_file)):
            with open(self.timing_file, 'r') as f:
                content = f.readlines()
            content = [x.strip() for x in content] 
            count = 0
            for l in content:
                count += 1
                d = json.loads(l)
                if(self.evaluate):
                    x.add_row([str('%3d' %count), d['time'], d['tag'], d['info'], d['file_read'], d['preprocess'], d['prediction'], d['demix'], d['eval'], d['total']])
                else:
                    x.add_row([str('%3d' %count), d['time'], d['tag'], d['info'], d['file_read'], d['preprocess'], d['prediction'], d['demix'], d['total']])
            print(x)
        else:
            print("No files executed in this run")

    def check_evaluate(self):
        if(self.evaluate == True and self.is_batch == True):
            print("Temp:", self.evaluate_path)
            evaluate_all(self.evaluate_path)


def get_args(): 
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file',action='store',type=str)
    group.add_argument('--batch',action='store_true')
    parser.add_argument('--new-project', action='store_true', required=False)
    parser.add_argument('--motion-correct-only', action='store_true', required=False)
    parser.add_argument('--evaluate', action='store_true', required=False)
    args = parser.parse_args()

    if(args.batch == True):
        task = 'batch'
    else:
        task = args.file

    p = pipeline_info(task, args.new_project, args.motion_correct_only, args.evaluate)

    return p




def main():

    tic = time.time()

    # Get the pipeline execution paramters and 
    # perform necessary initlizations
    p = get_args()

    # Execute the project
    if(p.is_batch == True):
        p.run_batch()
    else:
        p.run_file()

    # Check and run evaluation is required
    p.check_evaluate()

    # Print the timing information
    p.print_pretty()
    
    time_total = time.time() - tic
    printd("Total time:", time_total)

if __name__ == '__main__':
    main()