import random
import numpy as np
import open3d as o3d


def readpcd(path, rtype='pcd'):
    assert rtype in ['pcd', 'npy']
    pcd = o3d.io.read_point_cloud(path)
    if rtype == 'pcd':
        return pcd
    npy = np.asarray(pcd.points).astype(np.float32)
    return npy


def npy2pcd(npy, ind=-1):
    colors = [[1.0, 0, 0],
              [0, 1.0, 0],
              [0, 0, 1.0]]
    color = colors[ind] if ind < 3 else [random.random() for _ in range(3)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(npy)
    if ind >= 0:
        pcd.paint_uniform_color(color)
    return pcd


def pcd2npy(pcd):
    npy = np.asarray(pcd.points)
    return npy