import ntpath
import numpy as np
import tifffile as tiff
from keras import models

from .sequence import VI_Sequence
from .data import get_training_data, get_inference_data


def predict_and_merge(model, data_seq, patch_shape,
                      input_paths, target_paths, out_dir, ref_dir):
    
    preds = model.predict(data_seq)
    
    Ys, Xs = data_seq.get_tile_pos()
    
    for i, paths in enumerate(input_paths):
        input_img = []
        for path in paths:
            input_img.append(tiff.imread(path))
        pred_img = np.zeros_like(input_img[0])
        pred_count = np.zeros_like(input_img[0])
        img_length = len(input_img[0])
        for j in range(img_length):
            for k, (ys, xs) in enumerate(zip(Ys, Xs)):
                ye = ys + patch_shape[0]
                xe = xs + patch_shape[1]
                idx = (i * img_length + j) * len(Ys) + k
                pred_img[j, ys:ye, xs:xe] += preds[idx, :, :, 0]
                pred_count[j, ys:ye, xs:xe] += 1
    
        pred_img = pred_img / pred_count
        fname = ntpath.basename(paths[0])
        tiff.imsave(out_dir + '/' + fname,
                    pred_img.astype('float32'), photometric='minisblack')
        
        # reference output for visual inspection
        video = np.zeros(pred_img.shape[:2] + (0,), dtype='float32')
        for img in input_img:
            video = np.append(video, img / np.max(img), axis=2)
        if(target_paths is not None):
            target_img = tiff.imread(target_paths[i])
            video = np.append(video, target_img, axis=2)
        video = np.append(video, pred_img, axis=2)
        tiff.imsave(ref_dir + '/' + fname, video, photometric='minisblack')


def validate_model(input_dir_list, target_dir, model_dir, out_dir, ref_dir,
                   seed, validation_ratio,
                   patch_shape, tile_strides, batch_size):

    data = get_training_data(input_dir_list, target_dir,
                             seed, validation_ratio)
    valid_input_paths = data[2]
    valid_target_paths = data[3]
    
    valid_seq = VI_Sequence(batch_size, patch_shape,
                            valid_input_paths, valid_target_paths,
                            tiled=True, tile_strides=tile_strides)

    model = models.load_model(model_dir + '/model.h5')

    predict_and_merge(model, valid_seq, patch_shape,
                      valid_input_paths, valid_target_paths, out_dir, ref_dir)


def apply_model(input_dir_list, model_dir, out_dir, ref_dir,
                patch_shape, tile_strides, batch_size):

    input_files = get_inference_data(input_dir_list)
    model = models.load_model(model_dir + '/model.h5')
    for paths in input_files:
        data_seq = VI_Sequence(batch_size, patch_shape,
                               [paths], None,
                               tiled=True, tile_strides=tile_strides)
        
        predict_and_merge(model, data_seq, patch_shape,
                          [paths], None, out_dir, ref_dir)
