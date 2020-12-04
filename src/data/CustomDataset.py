import numpy as np
import os
import torch
from torch.utils.data import Dataset

from utils import readpcd
from utils import pc_normalize, random_select_points, shift_point_cloud, \
    jitter_point_cloud, generate_random_rotation_matrix, \
    generate_random_tranlation_vector, transform


class CustomData(Dataset):
    def __init__(self, root, npts, train=True):
        super(CustomData, self).__init__()
        dirname = 'train_data' if train else 'val_data'
        path = os.path.join(root, dirname)
        self.train = train
        self.files = [os.path.join(path, item) for item in sorted(os.listdir(path))]
        self.npts = npts

    def __getitem__(self, item):
        file = self.files[item]
        ref_cloud = readpcd(file, rtype='npy')
        ref_cloud = random_select_points(ref_cloud, m=self.npts)
        ref_cloud = pc_normalize(ref_cloud)
        R, t = generate_random_rotation_matrix(-20, 20), \
               generate_random_tranlation_vector(-0.5, 0.5)
        src_cloud = transform(ref_cloud, R, t)
        if self.train:
            ref_cloud = jitter_point_cloud(ref_cloud)
            src_cloud = jitter_point_cloud(src_cloud)
        return ref_cloud, src_cloud, R, t

    def __len__(self):
        return len(self.files)