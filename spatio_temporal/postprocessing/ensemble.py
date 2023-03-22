import os
import argparse
from tqdm.auto import tqdm

import numpy as np
import scipy


def get_args():
    parser = argparse.ArgumentParser(
        description='ensemble predictions')
    parser.add_argument('--input_dir', type=str,
                        help='dir that contains multiple submission dir')
    parser.add_argument('--output_dir', type=str,
                        help='output dir of ensembled prediction')
    return parser.parse_args()


def main():
    print("Ensemble submissions")

    print("==> load data")
    args = get_args()
    input_dir = args.input_dir
    pred_fnames = process_fname(input_dir)
    results = load_data(pred_fnames)
    print(f"====> find {len(results[0])} predictions")

    print("==> start ensemble")
    args = get_args()
    votes = []
    for one_txt in tqdm(results):
        votes.append(voting(one_txt))

    print("==> writing output")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_fnames = get_txt_names(input_dir)
    output_fnames = [os.path.join(output_dir, fn) for fn in output_fnames]
    assert len(output_fnames) == len(votes)

    for one_fname, one_pred in zip(output_fnames, votes):
        write_predition(one_fname, one_pred)


def process_fname(dir_of_pred_dirs):
    pred_dirs = os.listdir(dir_of_pred_dirs)
    pred_dirs = [os.path.join(dir_of_pred_dirs, one_dir)
                 for one_dir in pred_dirs]

    txt_names = sorted(os.listdir(pred_dirs[0]))
    for one_dir in pred_dirs:
        file_names = sorted(os.listdir(one_dir))
        assert file_names == txt_names

    pred_fnames = [[os.path.join(one_dir, fn) for fn in txt_names]
                   for one_dir in pred_dirs]

    return pred_fnames


def read_labels(fname):
    with open(fname) as f:
        lines = f.readlines()
    lines = [int(x.strip()) for x in lines]
    return lines


def load_data(pred_fnames):
    # txt: points: dir
    results = []
    for idx_txt in range(len(pred_fnames[0])):
        sub_collect = []
        for idx_dir in range(len(pred_fnames)):
            fn = pred_fnames[idx_dir][idx_txt]
            sub_collect.append(np.loadtxt(fn, dtype=int))
        sub_collect = np.vstack(sub_collect)
        results.append(sub_collect)
    return results


def voting(txt_preds: np.array):
    """return larger one if equal"""
    mode, counts = scipy.stats.mode(-txt_preds, axis=0, keepdims=False)
    return -mode


def get_txt_names(dir_of_pred_dirs):
    one_pred_dir = os.listdir(dir_of_pred_dirs)[0]
    one_pred_dir = os.path.join(dir_of_pred_dirs, one_pred_dir)
    txt_names = sorted(os.listdir(one_pred_dir))
    return txt_names


def write_predition(file_path, prediction):
    with open(file_path, 'w') as out_file:
        for pred in prediction:
            out_file.write(str(pred) + '\n')


if __name__ == "__main__":
    main()
