import copy
import numpy as np
import open3d as o3d


def icp(source, target):
    threshold = 2
    trans_init = np.eye(4, dtype=np.float32)

    reg_p2p = o3d.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPoint(),
        o3d.registration.ICPConvergenceCriteria(max_iteration=30)
    )

    transformation = reg_p2p.transformation
    estimate = copy.deepcopy(source)
    estimate.transform(transformation)
    R, t = transformation[:3, :3], transformation[:3, 3]
    return R, t, estimate