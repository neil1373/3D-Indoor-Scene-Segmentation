# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.

import lib.datasets.scannet200 as scannet200

DATASETS = []


def add_datasets(module):
    DATASETS.extend([getattr(module, a)
                    for a in dir(module) if 'Dataset' in a])


add_datasets(scannet200)


def load_dataset(name):
    '''Creates and returns an instance of the datasets given its name.
    '''
    # Find the model class from its name
    mdict = {dataset.__name__: dataset for dataset in DATASETS}
    if name not in mdict:
        print('Invalid dataset index. Options are:')
        # Display a list of valid dataset names
        for dataset in DATASETS:
            print('\t* {}'.format(dataset.__name__))
        raise ValueError(f'Dataset {name} not defined')
    DatasetClass = mdict[name]

    return DatasetClass
