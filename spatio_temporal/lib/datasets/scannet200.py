import glob
import logging
import os
import random
import sys
from pathlib import Path
import pickle

import numpy as np
from scipy import spatial, ndimage, misc
import torch
import volumentations as V

from lib.constants.dataset_sets import HEAD_CATS_SCANNET_200, COMMON_CATS_SCANNET_200, TAIL_CATS_SCANNET_200
from lib.dataset import SparseVoxelizationDataset, DatasetPhase, str2datasetphase_type
from lib.pc_utils import read_plyfile, save_point_cloud
from lib.transforms import InstanceAugmentation
from lib.utils import read_txt, fast_hist, per_class_iu

from lib.constants.scannet_constants import (
    SCANNET_COLOR_MAP_20, CLASS_LABELS_20, VALID_CLASS_IDS_20,
    SCANNET_COLOR_MAP_200, CLASS_LABELS_200, VALID_CLASS_IDS_200, SCANNET_COLOR_MAP_LONG, CLASS_LABELS_200
)
from lib.datasets.preprocessing.utils import box_intersect

import MinkowskiEngine as ME


class BasicScannetVoxelizationDataset(SparseVoxelizationDataset):

    # Voxelization arguments
    CLIP_BOUND = None
    TEST_CLIP_BOUND = None
    VOXEL_SIZE = 0.05

    # Load constants for label ids
    SCANNET_COLOR_MAP = SCANNET_COLOR_MAP_20
    CLASS_LABELS = CLASS_LABELS_20
    VALID_CLASS_IDS = VALID_CLASS_IDS_20

    # Augmentation arguments
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi,
                                                                                          np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
    ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

    ROTATION_AXIS = 'z'
    LOCFEAT_IDX = 2
    # Will be converted to 20 as defined in IGNORE_LABELS.
    NUM_LABELS = max(SCANNET_COLOR_MAP_LONG.keys())
    IGNORE_LABELS = tuple(set(range(NUM_LABELS)) - set(VALID_CLASS_IDS))
    IS_FULL_POINTCLOUD_EVAL = True

    # If trainval.txt does not exist, copy train.txt and add contents from val.txt
    DATA_PATH_FILE = {
        DatasetPhase.Train: 'train.txt',
        # DatasetPhase.Val: 'val.txt',
        DatasetPhase.Val: 'train.txt',
        DatasetPhase.TrainVal: 'trainval.txt',
        DatasetPhase.Test: 'test.txt'
    }

    def __init__(self,
                 config,
                 prevoxel_transform=None,
                 input_transform=None,
                 target_transform=None,
                 augment_data=True,
                 elastic_distortion=False,
                 explicit_rotation=-1,
                 ignore_mask=255,
                 return_transformation=False,
                 cache=False,
                 merge=False,
                 phase=DatasetPhase.Train):
        self.ignore_mask = ignore_mask
        self.explicit_rotation = explicit_rotation
        self.return_transformation = return_transformation
        if isinstance(phase, str):
            phase = str2datasetphase_type(phase)
        # Use cropped rooms for train/val
        data_root = config.scannet_path
        if phase not in [DatasetPhase.Train, DatasetPhase.TrainVal]:
            self.CLIP_BOUND = self.TEST_CLIP_BOUND
        data_paths = read_txt(os.path.join(data_root, self.DATA_PATH_FILE[phase]))
        logging.info('Loading {}: {}'.format(self.__class__.__name__, self.DATA_PATH_FILE[phase]))
        super().__init__(
            data_paths,
            data_root=data_root,
            prevoxel_transform=prevoxel_transform,
            input_transform=input_transform,
            target_transform=target_transform,
            ignore_label=config.ignore_label,
            return_transformation=config.return_transformation,
            augment_data=augment_data,
            elastic_distortion=elastic_distortion,
            merge=merge,
            phase=phase,
            config=config)

        # Load instance sampling weights for instance based balancing
        self.instance_sampling_weights = np.ones(self.NUM_LABELS)
        instance_sampling_weights_path = config.scannet_path + '/' + config.instance_sampling_weights
        if os.path.isfile(instance_sampling_weights_path) and config.sample_tail_instances:
            with open(instance_sampling_weights_path, "rb") as input_file:
                instance_sampling_weights = pickle.load(input_file)
                print('Loaded instance sampling probability weights {}'.format(instance_sampling_weights_path))
            for cat_id, cat_value in instance_sampling_weights.items():
                if cat_id > 0:
                    mapped_id = self.label_map[cat_id]
                    self.instance_sampling_weights[mapped_id] = cat_value
                    cat_name = CLASS_LABELS_200[self.VALID_CLASS_IDS.index(cat_id)]
                    # only tail instances are sampled
                    assert cat_value == 0 or cat_name in TAIL_CATS_SCANNET_200

        self.instance_sampling_weights /= self.instance_sampling_weights.sum()
        # Precompute a mapping from ids to categories
        self.id2cat_name = {}
        for id, cat_name in zip(self.VALID_CLASS_IDS, self.CLASS_LABELS):
            self.id2cat_name[id] = cat_name
        
        # Load the bounding boxes from all instances of the dataset
        bb_path = config.scannet_path + '/' + config.bounding_boxes_path
        if os.path.isfile(bb_path):
            with open(bb_path, 'rb') as f:
                self.bounding_boxes = pickle.load(f)

        # Calculate head-common-tail ids
        self.head_ids = []
        self.common_ids = []
        self.tail_ids = []
        self.frequency_organized_cats = torch.zeros(self.NUM_LABELS, 3).bool()
        for scannet_id, scannet_cat in zip(self.VALID_CLASS_IDS, self.CLASS_LABELS):
            if scannet_cat in HEAD_CATS_SCANNET_200:
                self.head_ids += [self.label_map[scannet_id]]
                self.frequency_organized_cats[self.label_map[scannet_id], 0] = True
            elif scannet_cat in COMMON_CATS_SCANNET_200:
                self.common_ids += [self.label_map[scannet_id]]
                self.frequency_organized_cats[self.label_map[scannet_id], 1] = True
            elif scannet_cat in TAIL_CATS_SCANNET_200:
                self.tail_ids += [self.label_map[scannet_id]]
                self.frequency_organized_cats[self.label_map[scannet_id], 2] = True

    def add_instances_to_cloud(self, coords, feats, labels, scene_name, transformations):

        if self.config.is_train:
            phase = 'train'
        else:
            phase = 'val'

        coords = coords.astype(int)
        voxel_scale, trans_rot = transformations

        instance_folder = self.config.scannet_path + f'/train/{phase}_instances/'
        num_instances = self.config.num_instances_to_add
        samples = np.random.choice(self.VALID_CLASS_IDS, num_instances, p=self.instance_sampling_weights)
        scene_bbs = self.bounding_boxes[scene_name]

        # Get scene dimensions
        scene_maxes = np.amax(coords, axis=0)
        scene_mins = np.amin(coords, axis=0)
        scene_dims = scene_maxes - scene_mins + 1

        # Create height map of the scene
        height_map = np.zeros((scene_dims[0], scene_dims[1])) + scene_mins[2]
        def calculate_height(coord):
            height_map[coord[0], coord[1]] = max(coord[2], height_map[coord[0], coord[1]])
        mapped_coords = coords - [scene_mins[0], scene_mins[1], 0]
        [calculate_height(coord) for coord in mapped_coords]

        # Apply a max smoothing to the height map to fill holes
        filled_height_map = ndimage.maximum_filter(height_map, size=5)  # magic number with 2cm vx size = 10cm real

        # Add sample
        for sample in samples:
            sample_cat = self.id2cat_name[sample]
            cat_path = instance_folder + sample_cat
            file = cat_path + '/' + random.choice(os.listdir(cat_path))
            inst_coords, inst_feats, inst_labels, instance_ids, _ = self.load_ply_w_path(file, scene_name)
                
            # Voxelize instance too
            inst_coords, inst_feats, inst_labels, inverse_map = self.sparse_voxelizer.voxelize(
                inst_coords, inst_feats, inst_labels)

            # Get instance dimensions
            sample_maxes = np.amax(inst_coords, axis=0)
            sample_mins = np.amin(inst_coords, axis=0)
            sample_dims = sample_maxes - sample_mins + 1

            # Start finding a suitable location
            centroid = np.zeros(3, dtype=int)
            iter_num = 0
            while iter_num < self.config.max_instance_placing_iterations:

                # Add random BB
                random_x = random.randint(scene_mins[0], scene_maxes[0])
                random_y = random.randint(scene_mins[1], scene_maxes[1])
                height = float(filled_height_map[random_x - scene_mins[0], random_y - scene_mins[1]]) + 0  # Add some margin (+20cm)
                centroid = np.array([random_x, random_y, int(height + sample_dims[2] / 2.)])

                random_bb = np.array([centroid - (sample_dims / 2.0), centroid + (sample_dims / 2.0)])

                # Check intersection
                is_intersects = False
                for bb_dict in scene_bbs['instances']:
                    # Load and ransform BB
                    bb = np.copy(bb_dict['bb'])
                    homo_bb = np.hstack((bb, np.ones((bb.shape[0], 1), dtype=coords.dtype)))
                    bb = homo_bb @ voxel_scale.T[:, :3]

                    if box_intersect(bb, random_bb):
                        is_intersects = True
                        break

                if not is_intersects:
                    break

                iter_num += 1

            # Push point cloud to BB location
            inst_coords = inst_coords - np.mean(inst_coords, axis=0).astype(int) + centroid

            # Append new inputs
            coords = np.concatenate((coords, inst_coords))
            feats = np.concatenate((feats, inst_feats))
            labels = np.concatenate((labels, inst_labels))

        # Finally, apply rotation augmentation
        homo_coords = np.hstack((coords, np.ones((coords.shape[0], 1), dtype=coords.dtype)))
        coords_aug = np.floor(homo_coords @ trans_rot.T[:, :3])

        # Quantize again to remove touching points
        _, unique_map = ME.utils.sparse_quantize(coords_aug, return_index=True, ignore_label=self.config.ignore_label)
        coords_aug, feats, labels = coords_aug[unique_map], feats[unique_map], labels[unique_map]

        return coords_aug, feats, labels    
    
    def __getitem__(self, index):
        if self.explicit_rotation > 1:
            rotation_space = np.linspace(-np.pi,
                                         np.pi, self.explicit_rotation + 1)
            rotation_angle = rotation_space[index % self.explicit_rotation]
            index //= self.explicit_rotation
        else:
            rotation_angle = None

        coords, feats, labels, instance_ids, scene_name = self.load_ply_aug(index)
        scene_name = scene_name.split('/')[-1].split('.')[0]       
        if labels is not None:
            pointcloud = np.hstack((coords, feats, labels[:, None]))
        else:
            dummy_labels = np.zeros(len(coords))
            pointcloud = np.hstack((coords, feats, dummy_labels[:, None]))

        if self.PREVOXELIZE_VOXEL_SIZE is not None:
            inds = ME.SparseVoxelize(
                pointcloud[:, :3] / self.PREVOXELIZE_VOXEL_SIZE, return_index=True)
            pointcloud = pointcloud[inds]

        if self.elastic_distortion:
            pointcloud = self._augment_elastic_distortion(pointcloud)

        coords, feats, labels = self.convert_mat2cfl(pointcloud)
        coords -= coords.mean(0)

        if (random.random() < 0.80) and self.config.data_aug_3d:
            # random scale
            if random.random() < 0.5:
                scale_bias = np.random.uniform(low=0.7, high=1.3, size=(3,))
                aug_scale = V.Scale3d(bias=scale_bias, always_apply=True)
                aug_result_scale = aug_scale(points=coords, features=feats, labels=labels)
                coords = aug_result_scale["points"]
                feats = aug_result_scale["features"]
                labels = aug_result_scale["labels"]

            
            # other augmentation
            aug_result_other = self.augs_self(points=coords, features=feats, labels=labels)
            coords = aug_result_other["points"]
            feats = aug_result_other["features"]
            labels = aug_result_other["labels"]
            coords -= coords.mean(0)


        if (random.random() < 0.80) and self.merge:
            sampled_index = random.randint(0, self.__len__() - 1)

            pointcloud1, _ = self.load_ply(sampled_index)
            if self.PREVOXELIZE_VOXEL_SIZE is not None:
                inds = ME.SparseVoxelize(
                    pointcloud1[:, :3] / self.PREVOXELIZE_VOXEL_SIZE, return_index=True)
                pointcloud1 = pointcloud1[inds]
            if self.elastic_distortion:
                pointcloud1 = self._augment_elastic_distortion(pointcloud1)
            coords1, feats1, labels1 = self.convert_mat2cfl(pointcloud1)
            aug1 = self.augs_mix(points=coords1, features=feats1, labels=labels1)
            coords1, feats1, labels1 = aug1["points"], aug1["features"], aug1["labels"]
            # coords1, feats1, labels1 = self.prevoxel_transform(coords1, feats1, labels1)

            coords -= coords.mean(0)
            coords += np.random.uniform(coords.min(0), coords.max(0)) / 2
            coords1 -= coords1.mean(0)
            # coords1 += np.random.uniform(coords1.min(0), coords1.max(0)) / 2
            coords = np.concatenate((coords, coords1))
            feats = np.concatenate((feats, feats1))
            labels = np.concatenate((labels, labels1))
            coords -= coords.mean(0)
            '''
            if coords.shape[0] > 400000: # clipping some points to avoid out of memory
                sample_idx = random.sample(range(coords.shape[0]), 400000)
                coords = coords[sample_idx]
                feats = feats[sample_idx]
                labels = labels[sample_idx]
            '''

        # return_transformation=True is for instance augmentation
        coords, feats, labels, inverse_map, transformations = self.sparse_voxelizer.voxelize(
            coords, feats, labels,
            rotation_angle=rotation_angle,
            return_transformation=True)                    
        
        # Add balanced instances
        if self.config.sample_tail_instances:
            # Don't augment yet, but apply when everything is given
            coords, feats, labels = self.add_instances_to_cloud(coords, feats, labels, scene_name, transformations)

        if self.phase == DatasetPhase.Test:
            self.inverse_maps[index] = inverse_map
        
        # map labels not used for evaluation to ignore_label
        if self.input_transform is not None:
            coords, feats, labels = self.input_transform(coords, feats, labels)
        if self.target_transform is not None:
            coords, feats, labels = self.target_transform(coords, feats, labels)
        if self.IGNORE_LABELS is not None:
            mapper = lambda x: self.label_map[x]
            if labels.ndim == 1:
                labels = np.vectorize(mapper)(labels)
            else:
                labels[:, 0] = np.vectorize(mapper)(labels[:, 0])

        return_args = [coords, feats, labels]

        if self.return_transformation:
            return_args.append(transformations[1].astype(np.float32))

        if self.phase == DatasetPhase.Test:
            return_args.append(inverse_map)

        return tuple(return_args)

    def get_output_id(self, iteration):
        return '_'.join(Path(self.data_paths[iteration]).stem.split('_')[:2])

    def get_classnames(self):
        return self.CLASS_LABELS



class Scannet200VoxelizationDataset(BasicScannetVoxelizationDataset):
    # Load constants for label ids
    SCANNET_COLOR_MAP = SCANNET_COLOR_MAP_200
    CLASS_LABELS = CLASS_LABELS_200
    VALID_CLASS_IDS = VALID_CLASS_IDS_200

    NUM_LABELS = max(SCANNET_COLOR_MAP_LONG.keys()) + 1
    IGNORE_LABELS = tuple(set(range(NUM_LABELS)) - set(VALID_CLASS_IDS))

class Scannet200Voxelization2cmDataset(Scannet200VoxelizationDataset):
    VOXEL_SIZE = 0.02

class ScannetVoxelization2cmDataset(BasicScannetVoxelizationDataset):
    VOXEL_SIZE = 0.02
