import random


def _get_file_list(input_dir_list, target_dir=None):
    """
    Get lists of input and target file paths.

    Parameters
    ----------
    See get_training_data().

    Returns
    -------
    input_files : see get_inference_data().
    target_files : list of pathlib.Path
        List of paths to the target files in target_dir.

    """
    input_files = []
    target_files = []
    filenames = sorted(input_dir_list[0].glob('*.tif'))
    for filename in filenames:
        inputs = []
        for input_dir in input_dir_list:
            inputs.append(input_dir.joinpath(filename.name))
        input_files.append(inputs)
        if(target_dir is not None):
            target_files.append(target_dir.joinpath(filename.name))

    return input_files, target_files


def get_training_data(input_dir_list, target_dir, seed, validation_ratio):
    """
    Get lists of input and target file paths and split them into training
    data and validation data. As long as seed and validation_ratio are kept
    the same, this function returns the same results.

    Parameters
    ----------
    input_dir_list : list of pathlib.Path
        List of directory paths containing input files. Each directory path
        corresponds to one channel of the input.
    target_dir : pathlib.Path
        Directory path containing target files.
    seed : integer
        Random seed for shuffling data before splitting.
    validation_ratio : integer
        What fraction of the inputs are used for validation. If there are
        N inputs, N/validation_ratio of them will be used for validation,
        while the rest will be used for training.

    Returns
    -------
    train_inputs : list of list of pathlib.Path
        List of paths to input files for training. Each item in the list is
        a file list corresponding to multiple channels of the input.
    train_targets : list of pathlib.Path
        List of paths to target files for training.
    valid_inputs : list of list of pathlib.Path
        List of paths to input files for validation. Each item in the list is
        a file list corresponding to multiple channels of the input.
    valid_targets : list of pathlib.Path
        List of paths to target files for validation.

    """
    input_files, target_files = _get_file_list(input_dir_list, target_dir)
    random.Random(seed).shuffle(input_files)
    random.Random(seed).shuffle(target_files)
    num_validation = len(input_files) // validation_ratio
    train_inputs = input_files[:-num_validation]
    train_targets = target_files[:-num_validation]
    valid_inputs = input_files[-num_validation:]
    valid_targets = target_files[-num_validation:]

    return train_inputs, train_targets, valid_inputs, valid_targets


def get_inference_data(input_dir_list):
    """
    Get a list of input file paths on which to perform U-Net inference.

    Parameters
    ----------
    input_dir_list : list of pathlib.Path
        List of directory paths containing input files. Each directory path
        corresponds to one channel of the input.

    Returns
    -------
    input_files : list of list of pathlib.Path
        List of paths to input files. Each item in this list is a file list
        corresponding to multiple channels of the input. That is, each file
        path in the file list corresponds to one input file in one of the
        directories in input_dir_list. More specifically, input_files[i][j]
        is the i-th file in the directory input_dir_list[j].

    """
    input_files, _ = _get_file_list(input_dir_list)
    return input_files
