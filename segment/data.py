import os
import random
import glob
import ntpath


def _get_file_list(input_dir_list, target_dir=''):
    input_files = []
    target_files = []
    filenames = glob.glob(input_dir_list[0] + '/*.tif')
    filenames.sort()
    for filename in filenames:
        basename = ntpath.basename(filename)
        inputs = []
        for input_dir in input_dir_list:
            inputs.append(os.path.join(input_dir, basename))
        input_files.append(inputs)
        target_files.append(os.path.join(target_dir, basename))

    return input_files, target_files


def get_training_data(input_dir_list, target_dir, seed, validation_ratio):
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
    input_files, _ = _get_file_list(input_dir_list)
    return input_files
