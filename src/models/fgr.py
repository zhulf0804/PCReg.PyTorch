import copy
import open3d as o3d


def fpfh(pcd, normals):
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=64))
    return pcd_fpfh


def execute_fast_global_registration(source, target, source_fpfh, target_fpfh):
    distance_threshold = 0.01
    result = o3d.registration.registration_fast_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh,
        o3d.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    transformation = result.transformation
    estimate = copy.deepcopy(source)
    estimate.transform(transformation)
    R, t = transformation[:3, :3], transformation[:3, 3]
    return R, t, estimate


def fgr(source, target, src_normals, tgt_normals):
    source_fpfh = fpfh(source, src_normals)
    target_fpfh = fpfh(target, tgt_normals)
    R, t, estimate = execute_fast_global_registration(source=source,
                                                      target=target,
                                                      source_fpfh=source_fpfh,
                                                      target_fpfh=target_fpfh)
    return R, t, estimate
