import argparse
import numpy as np
import open3d as o3d
import random
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import ModelNet40
from models import Benchmark, IterativeBenchmark, icp, fgr
from utils import npy2pcd, pcd2npy
from metrics import compute_metrics, summary_metrics, print_metrics


def config_params():
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--root', required=True, help='the data path')
    parser.add_argument('--infer_npts', type=int, default=-1,
                        help='the points number of each pc for training')
    parser.add_argument('--mode', default='clean',
                        choices=['clean', 'partial', 'noise'],
                        help='training mode about data')
    parser.add_argument('--normal', action='store_true',
                        help='whether to use normal data')
    parser.add_argument('--niters', type=int, default=8,
                        help='iteration nums in one model forward')
    parser.add_argument('--gn', action='store_true',
                        help='whether to use group normalization')
    parser.add_argument('--checkpoint', default='',
                        help='the path to the trained checkpoint')
    parser.add_argument('--method', default='benchmark',
                        help='choice=[benchmark, icp, fgr, bm_icp]')
    parser.add_argument('--cuda', action='store_true',
                        help='whether to use the cuda')
    parser.add_argument('--show', action='store_true',
                        help='whether to visualize')
    args = parser.parse_args()
    return args


def evaluate_benchmark(args, test_loader):
    in_dim = 6 if args.normal else 3
    model = IterativeBenchmark(in_dim=in_dim, niters=args.niters, gn=args.gn)
    if args.cuda:
        model = model.cuda()
        model.load_state_dict(torch.load(args.checkpoint))
    else:
        model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu')))
    model.eval()

    dura = []
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = [], [], [], [], [], []
    with torch.no_grad():
        for i, (ref_cloud, src_cloud, gtR, gtt) in tqdm(enumerate(test_loader)):
            if args.cuda:
                ref_cloud, src_cloud, gtR, gtt = ref_cloud.cuda(), src_cloud.cuda(), \
                                                 gtR.cuda(), gtt.cuda()
            tic = time.time()
            R, t, pred_ref_cloud = model(src_cloud.permute(0, 2, 1).contiguous(), 
                    ref_cloud.permute(0, 2, 1).contiguous())
            toc = time.time()
            dura.append(toc - tic)
            cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
            cur_t_isotropic = compute_metrics(R, t, gtR, gtt)
            r_mse.append(cur_r_mse)
            r_mae.append(cur_r_mae)
            t_mse.append(cur_t_mse)
            t_mae.append(cur_t_mae)
            r_isotropic.append(cur_r_isotropic.cpu().detach().numpy())
            t_isotropic.append(cur_t_isotropic.cpu().detach().numpy())

            if args.show:
                ref_cloud = torch.squeeze(ref_cloud).cpu().numpy()
                src_cloud = torch.squeeze(src_cloud).cpu().numpy()
                pred_ref_cloud = torch.squeeze(pred_ref_cloud[-1]).cpu().numpy()
                pcd1 = npy2pcd(ref_cloud, 0)
                pcd2 = npy2pcd(src_cloud, 1)
                pcd3 = npy2pcd(pred_ref_cloud, 2)
                o3d.visualization.draw_geometries([pcd1, pcd2, pcd3])

    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
        summary_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)

    return dura, r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic


def evaluate_icp(args, test_loader):
    dura = []
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = [], [], [], [], [], []
    for i, (ref_cloud, src_cloud, gtR, gtt) in tqdm(enumerate(test_loader)):
        if args.cuda:
            ref_cloud, src_cloud, gtR, gtt = ref_cloud.cuda(), src_cloud.cuda(), \
                                             gtR.cuda(), gtt.cuda()

        ref_cloud = torch.squeeze(ref_cloud).cpu().numpy()
        src_cloud = torch.squeeze(src_cloud).cpu().numpy()

        tic = time.time()
        R, t, pred_ref_cloud = icp(npy2pcd(src_cloud), npy2pcd(ref_cloud))
        toc = time.time()
        R = torch.from_numpy(np.expand_dims(R, 0)).to(gtR)
        t = torch.from_numpy(np.expand_dims(t, 0)).to(gtt)
        dura.append(toc - tic)

        cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
        cur_t_isotropic = compute_metrics(R, t, gtR, gtt)
        r_mse.append(cur_r_mse)
        r_mae.append(cur_r_mae)
        t_mse.append(cur_t_mse)
        t_mae.append(cur_t_mae)
        r_isotropic.append(cur_r_isotropic.cpu().detach().numpy())
        t_isotropic.append(cur_t_isotropic.cpu().detach().numpy())

        if args.show:
            pcd1 = npy2pcd(ref_cloud, 0)
            pcd2 = npy2pcd(src_cloud, 1)
            pcd3 = pred_ref_cloud
            o3d.visualization.draw_geometries([pcd1, pcd2, pcd3])

    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
        summary_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)

    return dura, r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic


def evaluate_fgr(args, test_loader):
    dura = []
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = [], [], [], [], [], []
    for i, (ref_cloud, src_cloud, gtR, gtt) in tqdm(enumerate(test_loader)):
        if args.cuda:
            ref_cloud, src_cloud, gtR, gtt = ref_cloud.cuda(), src_cloud.cuda(), \
                                             gtR.cuda(), gtt.cuda()

        ref_points = torch.squeeze(ref_cloud).cpu().numpy()[:, :3]
        src_points = torch.squeeze(src_cloud).cpu().numpy()[:, :3]
        ref_normals = torch.squeeze(ref_cloud).cpu().numpy()[:, 3:]
        src_normals = torch.squeeze(src_cloud).cpu().numpy()[:, 3:]

        tic = time.time()
        R, t, pred_ref_cloud = fgr(source=npy2pcd(src_points),
                                   target=npy2pcd(ref_points),
                                   src_normals=src_normals,
                                   tgt_normals=ref_normals)
        toc = time.time()
        R = torch.from_numpy(np.expand_dims(R, 0)).to(gtR)
        t = torch.from_numpy(np.expand_dims(t, 0)).to(gtt)
        dura.append(toc - tic)

        cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
        cur_t_isotropic = compute_metrics(R, t, gtR, gtt)
        r_mse.append(cur_r_mse)
        r_mae.append(cur_r_mae)
        t_mse.append(cur_t_mse)
        t_mae.append(cur_t_mae)
        r_isotropic.append(cur_r_isotropic.cpu().detach().numpy())
        t_isotropic.append(cur_t_isotropic.cpu().detach().numpy())

        if args.show:
            pcd1 = npy2pcd(ref_points, 0)
            pcd2 = npy2pcd(src_points, 1)
            pcd3 = pred_ref_cloud
            o3d.visualization.draw_geometries([pcd1, pcd2, pcd3])

    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
        summary_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)

    return dura, r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic


def evaluate_benchmark_icp(args, test_loader):
    in_dim = 6 if args.normal else 3
    model = IterativeBenchmark(in_dim=in_dim, niters=args.niters, gn=args.gn)
    if args.cuda:
        model = model.cuda()
        model.load_state_dict(torch.load(args.checkpoint))
    else:
        model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device('cpu')))
    model.eval()

    dura = []
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = [], [], [], [], [], []
    with torch.no_grad():
        for i, (ref_cloud, src_cloud, gtR, gtt) in tqdm(enumerate(test_loader)):
            if args.cuda:
                ref_cloud, src_cloud, gtR, gtt = ref_cloud.cuda(), src_cloud.cuda(), \
                                                 gtR.cuda(), gtt.cuda()
            tic = time.time()
            R1, t1, pred_ref_cloud = model(src_cloud.permute(0, 2, 1).contiguous(),
                    ref_cloud.permute(0, 2, 1).contiguous())
            ref_cloud = torch.squeeze(ref_cloud).cpu().numpy()
            src_cloud_tmp = torch.squeeze(pred_ref_cloud[-1]).cpu().numpy()
            R2, t2, pred_ref_cloud = icp(npy2pcd(src_cloud_tmp), npy2pcd(ref_cloud))
            R2, t2 = torch.from_numpy(R2)[None, ...].to(R1), \
                     torch.from_numpy(t2)[None, ...].to(R1)
            R, t = R2 @ R1, torch.squeeze(R2 @ t1[:, :, None], dim=-1) + t2
            toc = time.time()
            dura.append(toc - tic)
            cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
            cur_t_isotropic = compute_metrics(R, t, gtR, gtt)
            r_mse.append(cur_r_mse)
            r_mae.append(cur_r_mae)
            t_mse.append(cur_t_mse)
            t_mae.append(cur_t_mae)
            r_isotropic.append(cur_r_isotropic.cpu().detach().numpy())
            t_isotropic.append(cur_t_isotropic.cpu().detach().numpy())

            if args.show:
                src_cloud = torch.squeeze(src_cloud).cpu().numpy()
                pcd1 = npy2pcd(ref_cloud, 0)
                pcd2 = npy2pcd(src_cloud, 1)
                pcd3 = pred_ref_cloud
                o3d.visualization.draw_geometries([pcd1, pcd2, pcd3])

    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
        summary_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)

    return dura, r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic


if __name__ == '__main__':
    seed = 222
    random.seed(seed)
    np.random.seed(seed)

    args = config_params()
    print(args)
    test_set = ModelNet40(root=args.root,
                          npts=args.infer_npts,
                          train=False,
                          normal=args.normal,
                          mode=args.mode)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    if args.method == 'benchmark':
        dura, r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
            evaluate_benchmark(args, test_loader)
        print_metrics(args.method,
                      dura, r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)
    elif args.method == 'icp':
        dura, r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
            evaluate_icp(args, test_loader)
        print_metrics(args.method, dura, r_mse, r_mae, t_mse, t_mae, r_isotropic,
                      t_isotropic)
    elif args.method == 'fgr':
        dura, r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
            evaluate_fgr(args, test_loader)
        print_metrics(args.method, dura, r_mse, r_mae, t_mse, t_mae, r_isotropic,
                      t_isotropic)
    elif args.method == 'bm_icp':
        dura, r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
            evaluate_benchmark_icp(args, test_loader)
        print_metrics(args.method, dura, r_mse, r_mae, t_mse, t_mae, r_isotropic,
                      t_isotropic)
    else:
        raise ValueError