import copy
import numpy as np
import open3d as o3d


def icp(source, target):
    max_correspondence_distance = 2 # 0.5 in RPM-Net
    init = np.eye(4, dtype=np.float32)
    estimation_method = o3d.registration.TransformationEstimationPointToPoint()

    reg_p2p = o3d.registration.registration_icp(
        source=source,
        target=target,
        init=init,
        max_correspondence_distance=max_correspondence_distance,
        estimation_method=estimation_method
    )

    transformation = reg_p2p.transformation
    estimate = copy.deepcopy(source)
    estimate.transform(transformation)
    R, t = transformation[:3, :3], transformation[:3, 3]
    return R, t, estimate