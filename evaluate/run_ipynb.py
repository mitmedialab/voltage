import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter
from pathlib import Path

from importlib.resources import read_text
from . import templates


def _run_ipynb(data, out_dir, basename):
    nb = nbformat.reads(data, nbformat.NO_CONVERT)
    ep = ExecutePreprocessor(timeout=None)
    ep.preprocess(nb)
    
    ipynb_file = out_dir + '/' + basename + '.ipynb'
    with open(ipynb_file, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    he = HTMLExporter()
    he.template_name = 'classic'
    (body, resources) = he.from_notebook_node(nb)

    html_file = out_dir + '/' + basename + '.html'
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(body)


def run_ipynb_evaluate_each(in_file, gt_file, img_file, out_dir):
    data = read_text(templates, 'evaluate_each.ipynb')
    data = data.replace('@@@IN_FILE', in_file)
    data = data.replace('@@@GT_FILE', gt_file)
    data = data.replace('@@@IMG_FILE', img_file)
    data = data.replace('@@@OUT_DIR', out_dir)
    _run_ipynb(data, out_dir, Path(in_file).stem)

def run_ipynb_evaluate_all(out_dir):
    data = read_text(templates, 'evaluate_all.ipynb')
    data = data.replace('@@@OUT_DIR', out_dir)
    _run_ipynb(data, out_dir, 'all');
