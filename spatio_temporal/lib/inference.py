import os
import logging

import numpy as np
import torch

from lib.utils import (
    Timer, get_prediction, get_torch_device, read_txt
)
from lib.constants.scannet_constants import INVERSE_SCANNET200_LABEL_MAP

import MinkowskiEngine as ME


def test(model, test_data_loader, config):
    results, inverse_maps = predict(model, test_data_loader, config)
    assert len(inverse_maps) == len(results)

    data_root = config.scannet_path
    pred_dir = config.save_pred_dir
    os.makedirs(pred_dir, exist_ok=True)
    pred_names = get_output_filenames(data_root, pred_dir)
    assert len(pred_names) == len(results)

    for fn, one_pred, imap in zip(pred_names, results, inverse_maps):
        write_one_file(fn, one_pred, imap)


def predict(model, data_loader, config):
    device = get_torch_device(config.is_cuda)
    dataset = data_loader.dataset
    global_timer, data_timer, iter_timer = Timer(), Timer(), Timer()

    logging.info('===> Start testing')

    global_timer.tic()
    data_iter = data_loader.__iter__()
    max_iter = len(data_loader)

    mapping_array = get_inverse_func()

    # Fix batch normalization running mean and std
    model.eval()

    # Clear cache (when run in val mode, cleanup training cache)
    torch.cuda.empty_cache()

    result = []
    inverse_indices = []

    with torch.no_grad():
        for iteration in range(max_iter):
            data_timer.tic()
            if config.return_transformation:
                coords, input, target, pointcloud, transformation = data_iter.next()
            else:
                coords, input, target, inverse_index = data_iter.next()
            
            inverse_indices.append(inverse_index.squeeze().numpy())

            # Preprocess input
            iter_timer.tic()

            if config.normalize_color:
                input[:, :3] = input[:, :3] / 255. - 0.5
            # sinput = ME.SparseTensor(input, coords).to(device)
            input = input.float().to(device)
            coords = coords.to(device)
            sinput = ME.SparseTensor(input, coords)

            # Feed forward
            inputs = (sinput,)
            soutput = model(*inputs)
            output = soutput.F

            pred_200 = get_prediction(dataset, output, target).int()
            pred_real = inverse_label_mapping(
                pred_200.cpu().numpy(), mapping_array)
            result.append(pred_real)

            if iteration % config.empty_cache_freq == 0:
                # Clear cache
                torch.cuda.empty_cache()

    global_time = global_timer.toc(False)

    logging.info("Finished test. Elapsed time: {:.4f}".format(global_time))

    # Explicit memory cleanup
    if hasattr(data_iter, 'cleanup'):
        data_iter.cleanup()

    return result, inverse_indices


def get_inverse_func():
    label_200 = np.array(list(INVERSE_SCANNET200_LABEL_MAP.keys()))
    label_real = np.array(list(INVERSE_SCANNET200_LABEL_MAP.values()))
    mapping_array = np.zeros(label_200.max() + 1, dtype=int)
    mapping_array[label_200] = label_real
    return mapping_array


def inverse_label_mapping(pred, mapping_array):
    return mapping_array[pred]


def get_output_filenames(data_root, pred_dir):
    file_names = read_txt(os.path.join(data_root, 'test.txt'))
    file_names = sorted(file_names)
    file_names = [os.path.basename(fn) for fn in file_names]
    file_names = [fn.replace('ply', 'txt') for fn in file_names]
    os.makedirs(pred_dir, exist_ok=True)
    pred_names = [os.path.join(pred_dir, fn) for fn in file_names]
    return pred_names


def get_inverse_maps(test_data_set):
    for _ in test_data_set:
        pass
    return test_data_set.inverse_maps


def write_one_file(file_path, prediction, imap):
    real_preds = prediction[imap]
    with open(file_path, 'w') as out_file:
        for pred in real_preds:
            out_file.write(str(pred) + '\n')
