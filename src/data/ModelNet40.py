import h5py
import numpy as np
import open3d as o3d
import os
import torch

from torch.utils.data import Dataset
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOR_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOR_DIR)
from utils import pc_normalize, random_select_points, shift_point_cloud, \
    jitter_point_cloud, generate_random_rotation_matrix, \
    generate_random_tranlation_vector, transform, random_crop


class ModelNet40(Dataset):
    def __init__(self, root, npts, train=True, normal=False, mode='clean'):
        super(ModelNet40, self).__init__()
        self.npts = npts
        self.train = train
        self.normal = normal
        self.mode = mode
        files = [os.path.join(root, 'ply_data_train{}.h5'.format(i))
                 for i in range(5)]
        if not train:
            files = [os.path.join(root, 'ply_data_test{}.h5'.format(i))
                     for i in range(2)]
        self.data, self.labels = self.decode_h5(files)

    def decode_h5(self, files):
        points, normal, label = [], [], []
        for file in files:
            f = h5py.File(file, 'r')
            cur_points = f['data'][:].astype(np.float32)
            cur_normal = f['normal'][:].astype(np.float32)
            cur_label = f['label'][:].astype(np.float32)
            points.append(cur_points)
            normal.append(cur_normal)
            label.append(cur_label)
        points = np.concatenate(points, axis=0)
        normal = np.concatenate(normal, axis=0)
        data = np.concatenate([points, normal], axis=-1).astype(np.float32)
        label = np.concatenate(label, axis=0)
        return data, label

    def compose(self, mode, item):
        ref_cloud = self.data[item, ...]
        R, t = generate_random_rotation_matrix(), generate_random_tranlation_vector()
        if mode == 'clean':
            ref_cloud = random_select_points(ref_cloud, m=self.npts)
            src_cloud_points = transform(ref_cloud[:, :3], R, t)
            src_cloud_normal = transform(ref_cloud[:, 3:], R)
            src_cloud = np.concatenate([src_cloud_points, src_cloud_normal],
                                       axis=-1)
            return src_cloud, ref_cloud, R, t
        elif mode == 'partial':
            source_cloud = random_select_points(ref_cloud, m=self.npts)
            ref_cloud = random_select_points(ref_cloud, m=self.npts)
            src_cloud_points = transform(source_cloud[:, :3], R, t)
            src_cloud_normal = transform(source_cloud[:, 3:], R)
            src_cloud = np.concatenate([src_cloud_points, src_cloud_normal],
                                       axis=-1)
            src_cloud = random_crop(src_cloud, p_keep=0.7)
            return src_cloud, ref_cloud, R, t
        elif mode == 'noise':
            source_cloud = random_select_points(ref_cloud, m=self.npts)
            ref_cloud = random_select_points(ref_cloud, m=self.npts)
            src_cloud_points = transform(source_cloud[:, :3], R, t)
            src_cloud_normal = transform(source_cloud[:, 3:], R)
            src_cloud = np.concatenate([src_cloud_points, src_cloud_normal],
                                       axis=-1)
            return src_cloud, ref_cloud, R, t
        else:
            raise NotImplementedError

    def __getitem__(self, item):
        src_cloud, ref_cloud, R, t = self.compose(mode=self.mode, item=item)
        if self.train or self.mode == 'noise' or self.mode == 'partial':
            ref_cloud[:, :3] = jitter_point_cloud(ref_cloud[:, :3])
            src_cloud[:, :3] = jitter_point_cloud(src_cloud[:, :3])
        if not self.normal:
            ref_cloud, src_cloud = ref_cloud[:, :3], src_cloud[:, :3]
        return ref_cloud, src_cloud, R, t

    def __len__(self):
        return len(self.data)