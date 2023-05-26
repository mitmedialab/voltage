import pathlib
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter

from importlib.resources import read_text
from . import templates


def _run_ipynb(data, out_dir, out_basename):
    """
    Run Jupyter Notebook data and save the results as .ipynb and .html.

    Parameters
    ----------
    data : json
        Jupyter Notebook data.
    out_dir : string or pathlib.Path
        Directory path in which the results will be saved.
    out_basename : string
        File basename used to save the results.

    Returns
    -------
    None.

    """
    out_dir = pathlib.Path(out_dir)

    nb = nbformat.reads(data, nbformat.NO_CONVERT)
    ep = ExecutePreprocessor(timeout=None)
    ep.preprocess(nb)
    
    ipynb_file = out_dir.joinpath(out_basename + '.ipynb')
    with open(ipynb_file, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    he = HTMLExporter()
    he.template_name = 'classic'
    (body, resources) = he.from_notebook_node(nb)

    html_file = out_dir.joinpath(out_basename + '.html')
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(body)


def run_ipynb_evaluate_each(in_file, gt_file, img_file, spike_file,
                            out_dir, name):
    """
    Run the Jupyter Notebook for single file evaluation.

    Parameters
    ----------
    in_file : string or pathlib.Path
        Path to the input file containing the cell masks to be evaluated.
    gt_file : string or pathlib.Path
        Path to the ground truth cell masks.
    img_file : string or pathlib.Path
        Path to the representative image of the data set.
    spike_file : string or pathlib.Path
        Path to the file containing extracted voltage traces and spikes.
    out_dir : string or pathlib.Path
        Directory path in which the results will be saved.
    name : string
        Data set name used to show and save the results.

    Returns
    -------
    None.

    """
    data = read_text(templates, 'evaluate_each.ipynb')
    data = data.replace('@@@DATA_NAME', name)
    data = data.replace('@@@IN_FILE', str(in_file))
    data = data.replace('@@@GT_FILE', str(gt_file))
    data = data.replace('@@@IMG_FILE', str(img_file))
    data = data.replace('@@@SPIKE_FILE', str(spike_file))
    data = data.replace('@@@OUT_DIR', str(out_dir))
    _run_ipynb(data, out_dir, name)


def run_ipynb_evaluate_all(out_dir):
    """
    Run the Jupyter Notebook for overall evaluation.

    Parameters
    ----------
    out_dir : string or pathlib.Path
        Directory path in which the results will be saved.

    Returns
    -------
    None.

    """
    data = read_text(templates, 'evaluate_all.ipynb')
    data = data.replace('@@@OUT_DIR', str(out_dir))
    _run_ipynb(data, out_dir, 'all');
