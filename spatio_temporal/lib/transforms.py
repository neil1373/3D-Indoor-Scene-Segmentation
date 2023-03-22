# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu). All Rights Reserved.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part of
# the code.
import random

import logging
import numpy as np
import scipy
import scipy.ndimage
import scipy.interpolate
import torch
import MinkowskiEngine as ME

##############################
# Coordinate transformations
##############################
class InstanceAugmentation(object):
    def __init__(self, config):

        self.config = config

        self.rgb_to_hsv = HueSaturationTranslation.rgb_to_hsv
        self.hsv_to_rgb = HueSaturationTranslation.hsv_to_rgb

        # Color parameters
        self.color_shifts = ['Red', 'Green', 'Blue', 'Yellow', 'Dark', 'Bright']
        self.red_hue = 0. / 360.0
        self.yellow_hue = 60. / 360.0
        self.green_hue = 120. / 360.0
        self.blue_hue = 240. / 360.0
        self.white_scale = 2.0
        self.black_scale = 1. / self.white_scale

        # Scale parameters
        self.size_shifts = [0.5, 1.5]


    def shift_hue(self, colors, h_out):
        hsv = self.rgb_to_hsv(colors / 255.)
        hsv[..., 0] = h_out
        rgb = self.hsv_to_rgb(hsv) * 255.
        return rgb


    def shift_color(self, coords, feats, labels):
        color_direction = random.sample(self.color_shifts, 1)[0]

        if color_direction == 'Red':
            feats = self.shift_hue(feats, self.red_hue)
            labels[:, 1] = 1
        elif color_direction == 'Green':
            feats = self.shift_hue(feats, self.green_hue)
            labels[:, 1] = 2
        elif color_direction == 'Blue':
            feats = self.shift_hue(feats, self.blue_hue)
            labels[:, 1] = 3
        elif color_direction == 'Yellow':
            feats = self.shift_hue(feats, self.yellow_hue)
            labels[:, 1] = 4
        elif color_direction == 'Dark':
            feats = (feats * self.black_scale).astype(int)
            labels[:, 1] = 5
        elif color_direction == 'Bright':
            diff_to_max = 255 - feats
            feats = (255 - (diff_to_max / self.white_scale)).astype(int)
            labels[:, 1] = 6

        return coords, feats, labels


    def shift_scale(self, coords, feats, labels, scene_scale):

        scale_direction = np.random.uniform(low=0., high=2.)

        # Upsample for scaling up
        if scale_direction > 1.:

            # pick random positive scaling factor
            inst_scale = np.array([(coords[:, 0].max() - coords[:, 0].min()),
                                   (coords[:, 1].max() - coords[:, 1].min()),
                                   (coords[:, 2].max() - coords[:, 2].min())])
            scale_direction = np.random.uniform(low=1.0, high=min(self.size_shifts[1], (scene_scale/inst_scale).min()))

            # scale and center to same centroid in X-Y, push up in Z
            center_x = (coords[:, 0].min() + coords[:, 0].max()) / 2.
            center_y = (coords[:, 1].min() + coords[:, 1].max()) / 2.
            min_z = coords[:, 2].min()

            coords *= scale_direction
            coords += np.array([center_x, center_y, min_z]) * (1 - scale_direction)

            labels = np.ones((coords.shape[0], 2)) * labels[0, 0]
            labels[:, 1] = 7
            return coords, feats, labels

        elif scale_direction <= 1.:

            # pick random negative scaling factor
            scale_direction = np.random.uniform(low=self.size_shifts[0], high=1.0)

            # scale and center to same centroid in X-Y, push up in Z
            center_x = (coords[:, 0].min() + coords[:, 0].max()) / 2.
            center_y = (coords[:, 1].min() + coords[:, 1].max()) / 2.
            min_z = coords[:, 2].min()

            coords *= scale_direction
            coords += np.array([center_x, center_y, min_z]) * (1 - scale_direction)

            labels[:, 1] = 8
            return coords, feats, labels

class RandomDropout(object):

  def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.5):
    """
    upright_axis: axis index among x,y,z, i.e. 2 for z
    """
    self.dropout_ratio = dropout_ratio
    self.dropout_application_ratio = dropout_application_ratio

  def __call__(self, coords, feats, labels):
    if random.random() < self.dropout_ratio:
      N = len(coords)
      inds = np.random.choice(N, int(N * (1 - self.dropout_ratio)), replace=False)
      return coords[inds], feats[inds], labels[inds]
    return coords, feats, labels


class RandomHorizontalFlip(object):

  def __init__(self, upright_axis, is_temporal):
    """
    upright_axis: axis index among x,y,z, i.e. 2 for z
    """
    self.is_temporal = is_temporal
    self.D = 4 if is_temporal else 3
    self.upright_axis = {'x': 0, 'y': 1, 'z': 2}[upright_axis.lower()]
    # Use the rest of axes for flipping.
    self.horz_axes = set(range(self.D)) - set([self.upright_axis])

  def __call__(self, coords, feats, labels):
    if random.random() < 0.95:
      for curr_ax in self.horz_axes:
        if random.random() < 0.5:
          coord_max = np.max(coords[:, curr_ax])
          coords[:, curr_ax] = coord_max - coords[:, curr_ax]
    return coords, feats, labels


class ChromaticTranslation(object):
  """Add random color to the image, input must be an array in [0,255] or a PIL image"""

  def __init__(self, trans_range_ratio=1e-1):
    """
    trans_range_ratio: ratio of translation i.e. 255 * 2 * ratio * rand(-0.5, 0.5)
    """
    self.trans_range_ratio = trans_range_ratio

  def __call__(self, coords, feats, labels):
    if random.random() < 0.95:
      tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.trans_range_ratio
      feats[:, :3] = np.clip(tr + feats[:, :3], 0, 255)
    return coords, feats, labels


class ChromaticAutoContrast(object):

  def __init__(self, randomize_blend_factor=True, blend_factor=0.5):
    self.randomize_blend_factor = randomize_blend_factor
    self.blend_factor = blend_factor

  def __call__(self, coords, feats, labels):
    if random.random() < 0.2:
      # mean = np.mean(feats, 0, keepdims=True)
      # std = np.std(feats, 0, keepdims=True)
      # lo = mean - std
      # hi = mean + std
      lo = np.min(feats, 0, keepdims=True)
      hi = np.max(feats, 0, keepdims=True)

      scale = 255 / (hi - lo)

      contrast_feats = (feats - lo) * scale

      blend_factor = random.random() if self.randomize_blend_factor else self.blend_factor
      feats = (1 - blend_factor) * feats + blend_factor * contrast_feats
    return coords, feats, labels


class ChromaticJitter(object):

  def __init__(self, std=0.01):
    self.std = std

  def __call__(self, coords, feats, labels):
    if random.random() < 0.95:
      noise = np.random.randn(feats.shape[0], 3)
      noise *= self.std * 255
      feats[:, :3] = np.clip(noise + feats[:, :3], 0, 255)
    return coords, feats, labels


class HueSaturationTranslation(object):

  @staticmethod
  def rgb_to_hsv(rgb):
    # Translated from source of colorsys.rgb_to_hsv
    # r,g,b should be a numpy arrays with values between 0 and 255
    # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
    rgb = rgb.astype('float')
    hsv = np.zeros_like(rgb)
    # in case an RGBA array was passed, just copy the A channel
    hsv[..., 3:] = rgb[..., 3:]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb[..., :3], axis=-1)
    minc = np.min(rgb[..., :3], axis=-1)
    hsv[..., 2] = maxc
    mask = maxc != minc
    hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
    gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
    bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
    hsv[..., 0] = np.select([r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
    hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
    return hsv

  @staticmethod
  def hsv_to_rgb(hsv):
    # Translated from source of colorsys.hsv_to_rgb
    # h,s should be a numpy arrays with values between 0.0 and 1.0
    # v should be a numpy array with values between 0.0 and 255.0
    # hsv_to_rgb returns an array of uints between 0 and 255.
    rgb = np.empty_like(hsv)
    rgb[..., 3:] = hsv[..., 3:]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).astype('uint8')
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
    rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
    rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
    rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
    return rgb.astype('uint8')

  def __init__(self, hue_max, saturation_max):
    self.hue_max = hue_max
    self.saturation_max = saturation_max

  def __call__(self, coords, feats, labels):
    # Assume feat[:, :3] is rgb
    hsv = HueSaturationTranslation.rgb_to_hsv(feats[:, :3])
    hue_val = (random.random() - 0.5) * 2 * self.hue_max
    sat_ratio = 1 + (random.random() - 0.5) * 2 * self.saturation_max
    hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
    hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
    feats[:, :3] = np.clip(HueSaturationTranslation.hsv_to_rgb(hsv), 0, 255)

    # pcd = o3d.PointCloud()
    # pcd.points = o3d.Vector3dVector(coords)
    # pcd.colors = o3d.Vector3dVector(feats / 255)
    # o3d.draw_geometries([pcd])

    return coords, feats, labels


class HeightTranslation(object):

  def __init__(self, std=0.01):
    self.std = std

  def __call__(self, coords, feats, labels):
    if feats.shape[1] > 3 and random.random() < 0.95:
      feats[:, -1] += np.random.randn(1) * self.std
    return coords, feats, labels


class HeightJitter(object):

  def __init__(self, std):
    self.std = std

  def __call__(self, coords, feats, labels):
    if feats.shape[1] > 3 and random.random() < 0.95:
      feats[:, -1] += np.random.randn(feats.shape[0]) * self.std
    return coords, feats, labels


class NormalJitter(object):

  def __init__(self, std):
    self.std = std

  def __call__(self, coords, feats, labels):
    # normal jitter
    if feats.shape[1] > 6 and random.random() < 0.95:
      feats[:, 3:6] += np.random.randn(feats.shape[0], 3) * self.std
    return coords, feats, labels


class Compose(object):
  """Composes several transforms together."""

  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, *args):
    for t in self.transforms:
      args = t(*args)
    return args


class cfl_collate_fn_factory:
  """Generates collate function for coords, feats, labels.

    Args:
      limit_numpoints: If 0 or False, does not alter batch size. If positive integer, limits batch
                       size so that the number of input coordinates is below limit_numpoints.
  """

  def __init__(self, limit_numpoints):
    self.limit_numpoints = limit_numpoints

  def __call__(self, list_data):
    coords, feats, labels = list(zip(*list_data))
    coords_batch, feats_batch, labels_batch = [], [], []

    batch_id = 0
    batch_num_points = 0
    for batch_id, _ in enumerate(coords):
      num_points = coords[batch_id].shape[0]
      batch_num_points += num_points
      if self.limit_numpoints and batch_num_points > self.limit_numpoints:
        num_full_points = sum(len(c) for c in coords)
        num_full_batch_size = len(coords)
        logging.warning(
            f'\t\tCannot fit {num_full_points} points into {self.limit_numpoints} points '
            f'limit. Truncating batch size at {batch_id} out of {num_full_batch_size} with {batch_num_points - num_points}.'
        )
        break
      coords_batch.append(torch.from_numpy(coords[batch_id]).int())
      feats_batch.append(torch.from_numpy(feats[batch_id]))
      labels_batch.append(torch.from_numpy(labels[batch_id]).int())

    # Concatenate all lists
    coords_batch, feats_batch, labels_batch = ME.utils.sparse_collate(coords_batch, feats_batch, labels_batch)
    return coords_batch, feats_batch, labels_batch


class cfli_collate_fn_factory:
  """Generates collate function for coords, feats, labels, inverse_index.

    Args:
      limit_numpoints: If 0 or False, does not alter batch size. If positive integer, limits batch
                       size so that the number of input coordinates is below limit_numpoints.
  """

  def __init__(self, limit_numpoints):
    self.limit_numpoints = limit_numpoints

  def __call__(self, list_data):
    coords, feats, labels, inverse_idx = list(zip(*list_data))
    coords_batch, feats_batch, labels_batch, inverse_idx_batch = [], [], [], []

    batch_id = 0
    batch_num_points = 0
    for batch_id, _ in enumerate(coords):
      num_points = coords[batch_id].shape[0]
      batch_num_points += num_points
      if self.limit_numpoints and batch_num_points > self.limit_numpoints:
        num_full_points = sum(len(c) for c in coords)
        num_full_batch_size = len(coords)
        logging.warning(
            f'\t\tCannot fit {num_full_points} points into {self.limit_numpoints} points '
            f'limit. Truncating batch size at {batch_id} out of {num_full_batch_size} with {batch_num_points - num_points}.'
        )
        break
      coords_batch.append(torch.from_numpy(coords[batch_id]).int())
      feats_batch.append(torch.from_numpy(feats[batch_id]))
      labels_batch.append(torch.from_numpy(labels[batch_id]).int())
      inverse_idx_batch.append(torch.from_numpy(inverse_idx[batch_id]).int())

    # Concatenate all lists
    coords_batch, feats_batch, labels_batch = ME.utils.sparse_collate(coords_batch, feats_batch, labels_batch)
    inverse_idx_batch = torch.cat(inverse_idx_batch, 0)
    return coords_batch, feats_batch, labels_batch, inverse_idx_batch
class cflt_collate_fn_factory:
  """Generates collate function for coords, feats, labels, point_clouds, transformations.

    Args:
      limit_numpoints: If 0 or False, does not alter batch size. If positive integer, limits batch
                       size so that the number of input coordinates is below limit_numpoints.
  """

  def __init__(self, limit_numpoints):
    self.limit_numpoints = limit_numpoints

  def __call__(self, list_data):
    coords, feats, labels, pointclouds, transformations = list(zip(*list_data))
    cfl_collate_fn = cfl_collate_fn_factory(limit_numpoints=self.limit_numpoints)
    coords_batch, feats_batch, labels_batch = cfl_collate_fn(list(zip(coords, feats, labels)))

    batch_id = 0
    batch_num_points = 0
    pointclouds_batch, transformations_batch = [], []
    for pointcloud, transformation in zip(pointclouds, transformations):
      num_points = len(pointcloud)
      batch_num_points += num_points
      if self.limit_numpoints and batch_num_points > self.limit_numpoints:
        break

      pointclouds_batch.append(
          torch.cat((torch.ones(pointcloud.shape[0], 1) * batch_id, torch.from_numpy(pointcloud)), 1))
      transformations_batch.append(
          torch.cat((torch.ones(transformation.shape[0], 1) * batch_id, torch.from_numpy(transformation)), 1))

      batch_id += 1

    pointclouds_batch = torch.cat(pointclouds_batch, 0).float()
    transformations_batch = torch.cat(transformations_batch, 0).float()
    return coords_batch, feats_batch, labels_batch, pointclouds_batch, transformations_batch


def elastic_distortion(pointcloud, granularity, magnitude):
  """Apply elastic distortion on sparse coordinate space.

    pointcloud: numpy array of (number of points, at least 3 spatial dims)
    granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
    magnitude: noise multiplier
  """
  blurx = np.ones((3, 1, 1, 1)).astype('float32') / 3
  blury = np.ones((1, 3, 1, 1)).astype('float32') / 3
  blurz = np.ones((1, 1, 3, 1)).astype('float32') / 3
  coords = pointcloud[:, :3]
  coords_min = coords.min(0)

  # Create Gaussian noise tensor of the size given by granularity.
  noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
  noise = np.random.randn(*noise_dim, 3).astype(np.float32)

  # Smoothing.
  for _ in range(2):
    noise = scipy.ndimage.filters.convolve(noise, blurx, mode='constant', cval=0)
    noise = scipy.ndimage.filters.convolve(noise, blury, mode='constant', cval=0)
    noise = scipy.ndimage.filters.convolve(noise, blurz, mode='constant', cval=0)

  # Trilinear interpolate noise filters for each spatial dimensions.
  ax = [
      np.linspace(d_min, d_max, d)
      for d_min, d_max, d in zip(coords_min - granularity, coords_min + granularity *
                                 (noise_dim - 2), noise_dim)
  ]
  interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
  pointcloud[:, :3] = coords + interp(coords) * magnitude
  return pointcloud
