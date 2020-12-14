import time
import json
import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter
import pathlib
import glob

def evaluate_each(fname, outdir_eval):
    with open('python/evaluate_each.ipynb') as f:
        data = f.read()
    data = data.replace('@@@FNAME', fname)
    nb = nbformat.reads(data, nbformat.NO_CONVERT)
    basename = pathlib.Path(fname).stem
    ep = ExecutePreprocessor(timeout=None)
    ep.preprocess(nb)
    with open(outdir_eval + '/' + basename + '.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    he = HTMLExporter()
    he.template_name = 'classic'
    (body, resources) = he.from_notebook_node(nb)
    with open(outdir_eval + '/' + basename + '.html', 'w', encoding='utf-8') as f:
        f.write(body)

def run_evaluation_file_background():

    with open('settings.txt') as file:
        settings = json.load(file)
    EVAL_PATH = settings['output_base_path'] + '/' + settings['evaluation_result_path'] + '/'
    EXECUTION_INFO_FILE = settings['output_base_path'] + '/' + settings['log_path'] + '/' + 'execution.info'


    with open('file_params.txt') as file:
        params = json.load(file)

    processed_file_list = []

    while True:
        try:
            with open(EXECUTION_INFO_FILE, 'r') as f:
                content = f.readlines()
        except:
            time.sleep(1)
            continue
            
        content = [x.strip() for x in content] 
        if(len(content) == 0):
            time.sleep(1)

        for l in content:
            d = json.loads(l)
            if(d['tag'] == 'None'):
                if(d['msg'] == 'FinishedExecution'):
                    return
            if(d['tag'] not in processed_file_list):
                processed_file_list.append(d['tag'])
                tag = d['tag']
                fname = params[d['tag']]['filename']
                evaluate_each(fname, EVAL_PATH)
            else:
                time.sleep(1)

if __name__ == '__main__':
    run_evaluation_file_background()