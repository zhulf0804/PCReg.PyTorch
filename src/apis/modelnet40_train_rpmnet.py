import argparse
import numpy as np
import open3d
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Dict, List

from data import ModelNet40
from models import IterativeBenchmark, RPMNet
from loss import EMDLosspy
from metrics import compute_metrics, summary_metrics, print_train_info
from utils import time_calc, inv_R_t


def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def config_params():
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    ## dataset
    parser.add_argument('--root', required=True, help='the data path')
    parser.add_argument('--train_npts', type=int, default=1024,
                        help='the points number of each pc for training')
    parser.add_argument('--normal', action='store_true',
                        help='whether to use normal data')
    parser.add_argument('--mode', default='clean',
                        choices=['clean', 'partial', 'noise'],
                        help='training mode about data')
    ## models training
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--epoches', type=int, default=400)
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--radius', type=float, default=0.3,
                        help='Neighborhood radius for computing pointnet features')
    parser.add_argument('--num_neighbors', type=int, default=64, metavar='N',
                        help='Max num of neighbors to use')
    # RPMNet settings
    parser.add_argument('--feat_dim', type=int, default=96,
                        help='Feature dimension (to compute distances on). Other numbers will be scaled accordingly')
    parser.add_argument('--no_slack', action='store_true',
                        help='If set, will not have a slack column.')
    parser.add_argument('--num_sk_iter', type=int, default=5,
                        help='Number of inner iterations used in sinkhorn normalization')
    parser.add_argument('--milestones', type=list, default=[50, 250],
                        help='lr decays when epoch in milstones')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='lr decays to gamma * lr every decay epoch')
    # logs
    parser.add_argument('--saved_path', default='work_dirs/models',
                        help='the path to save training logs and checkpoints')
    parser.add_argument('--saved_frequency', type=int, default=10,
                        help='the frequency to save the logs and checkpoints')
    args = parser.parse_args()
    return args


def transform_func(g, a, normals=None):
    """ Applies the SE3 transform

    Args:
        g: SE3 transformation matrix of size ([1,] 3/4, 4) or (B, 3/4, 4)
        a: Points to be transformed (N, 3) or (B, N, 3)
        normals: (Optional). If provided, normals will be transformed

    Returns:
        transformed points of size (N, 3) or (B, N, 3)

    """
    R = g[..., :3, :3]  # (B, 3, 3)
    p = g[..., :3, 3]  # (B, 3)

    if len(g.size()) == len(a.size()):
        b = torch.matmul(a, R.transpose(-1, -2)) + p[..., None, :]
    else:
        raise NotImplementedError
        b = R.matmul(a.unsqueeze(-1)).squeeze(-1) + p  # No batch. Not checked

    if normals is not None:
        rotated_normals = normals @ R.transpose(-1, -2)
        return b, rotated_normals

    else:
        return b


def compute_losses(data: Dict, pred_transforms: List, endpoints: Dict,
                   loss_type: str = 'mae', reduction: str = 'mean') -> Dict:
    """Compute losses

    Args:
        data: Current mini-batch data
        pred_transforms: Predicted transform, to compute main registration loss
        endpoints: Endpoints for training. For computing outlier penalty
        loss_type: Registration loss type, either 'mae' (Mean absolute error, used in paper) or 'mse'
        reduction: Either 'mean' or 'none'. Use 'none' to accumulate losses outside
                   (useful for accumulating losses for entire validation dataset)

    Returns:
        losses: Dict containing various fields. Total loss to be optimized is in losses['total']

    """
    wt_inliers = 1e-2
    losses = {}
    num_iter = len(pred_transforms)

    # Compute losses
    gt_src_transformed = transform_func(data['transform_gt'], data['points_src'][..., :3])
    if loss_type == 'mse':
        # MSE loss to the groundtruth (does not take into account possible symmetries)
        criterion = nn.MSELoss(reduction=reduction)
        for i in range(num_iter):
            pred_src_transformed = se3.transform(pred_transforms[i], data['points_src'][..., :3])
            if reduction.lower() == 'mean':
                losses['mse_{}'.format(i)] = criterion(pred_src_transformed, gt_src_transformed)
            elif reduction.lower() == 'none':
                losses['mse_{}'.format(i)] = torch.mean(criterion(pred_src_transformed, gt_src_transformed),
                                                        dim=[-1, -2])
    elif loss_type == 'mae':
        # MSE loss to the groundtruth (does not take into account possible symmetries)
        criterion = nn.L1Loss(reduction=reduction)
        for i in range(num_iter):
            pred_src_transformed = transform_func(pred_transforms[i], data['points_src'][..., :3])
            if reduction.lower() == 'mean':
                losses['mae_{}'.format(i)] = criterion(pred_src_transformed, gt_src_transformed)
            elif reduction.lower() == 'none':
                losses['mae_{}'.format(i)] = torch.mean(criterion(pred_src_transformed, gt_src_transformed),
                                                        dim=[-1, -2])
    else:
        raise NotImplementedError

    # Penalize outliers
    for i in range(num_iter):
        ref_outliers_strength = (1.0 - torch.sum(endpoints['perm_matrices'][i], dim=1)) * wt_inliers
        src_outliers_strength = (1.0 - torch.sum(endpoints['perm_matrices'][i], dim=2)) * wt_inliers
        if reduction.lower() == 'mean':
            losses['outlier_{}'.format(i)] = torch.mean(ref_outliers_strength) + torch.mean(src_outliers_strength)
        elif reduction.lower() == 'none':
            losses['outlier_{}'.format(i)] = torch.mean(ref_outliers_strength, dim=1) + \
                                             torch.mean(src_outliers_strength, dim=1)

    discount_factor = 0.5  # Early iterations will be discounted
    total_losses = []
    for k in losses:
        discount = discount_factor ** (num_iter - int(k[k.rfind('_')+1:]) - 1)
        total_losses.append(losses[k] * discount)
    losses['total'] = torch.sum(torch.stack(total_losses), dim=0)

    return losses['total']


@time_calc
def train_one_epoch(train_loader, model, loss_fn, optimizer):
    losses = []
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = [], [], [], [], [], []
    for ref_cloud, src_cloud, gtR, gtt in tqdm(train_loader):
        ref_cloud, src_cloud, gtR, gtt = ref_cloud.cuda(), src_cloud.cuda(), \
                                         gtR.cuda(), gtt.cuda()
        optimizer.zero_grad()
        pred_transforms, endpoints = model(xyz_ref=ref_cloud[..., :3],
                                           norm_ref=ref_cloud[..., 3:],
                                           xyz_src=src_cloud[..., :3],
                                           norm_src=src_cloud[..., 3:],
                                           num_iter=2)
        R, t = pred_transforms[-1][:, :3, :3], pred_transforms[-1][:, :3, 3]
        #loss = loss_fn(ref_cloud[..., :3].contiguous(),
        #               pred_ref_clouds[..., :3].contiguous())
        #loss = compute_loss(ref_cloud, pred_ref_clouds, loss_fn)
        data = {}
        inv_R, inv_t = inv_R_t(gtR, gtt)
        B, _, _ = inv_R.size()
        data['transform_gt'] = torch.zeros((B, 3, 4), dtype=torch.float32).to(inv_R)
        data['transform_gt'][:, :3, :3] = inv_R
        data['transform_gt'][:, :3, 3] = inv_t
        data['points_src'] = src_cloud
        loss = compute_losses(data, pred_transforms, endpoints)
        loss.backward()
        optimizer.step()

        cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
        cur_t_isotropic = compute_metrics(R, t, gtR, gtt)
        losses.append(loss.item())
        r_mse.append(cur_r_mse)
        r_mae.append(cur_r_mae)
        t_mse.append(cur_t_mse)
        t_mae.append(cur_t_mae)
        r_isotropic.append(cur_r_isotropic.cpu().detach().numpy())
        t_isotropic.append(cur_t_isotropic.cpu().detach().numpy())
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
        summary_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)
    results = {
        'loss': np.mean(losses),
        'r_mse': r_mse,
        'r_mae': r_mae,
        't_mse': t_mse,
        't_mae': t_mae,
        'r_isotropic': r_isotropic,
        't_isotropic': t_isotropic
    }
    return results


@time_calc
def test_one_epoch(test_loader, model, loss_fn):
    model.eval()
    losses = []
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = [], [], [], [], [], []
    with torch.no_grad():
        for ref_cloud, src_cloud, gtR, gtt in tqdm(test_loader):
            ref_cloud, src_cloud, gtR, gtt = ref_cloud.cuda(), src_cloud.cuda(), \
                                             gtR.cuda(), gtt.cuda()
            pred_transforms, endpoints = model(xyz_ref=ref_cloud[..., :3],
                                               norm_ref=ref_cloud[..., 3:],
                                               xyz_src=src_cloud[..., :3],
                                               norm_src=src_cloud[..., 3:],
                                               num_iter=5)
            R, t = pred_transforms[-1][:, :3, :3], pred_transforms[-1][:, :3, 3]
            # loss = loss_fn(ref_cloud[..., :3].contiguous(),
            #               pred_ref_clouds[..., :3].contiguous())
            # loss = compute_loss(ref_cloud, pred_ref_clouds, loss_fn)
            data = {}
            inv_R, inv_t = inv_R_t(gtR, gtt)
            B, _, _ = inv_R.size()
            data['transform_gt'] = torch.zeros((B, 3, 4),
                                               dtype=torch.float32).to(inv_R)
            data['transform_gt'][:, :3, :3] = inv_R
            data['transform_gt'][:, :3, 3] = inv_t
            data['points_src'] = src_cloud
            loss = compute_losses(data, pred_transforms, endpoints)

            cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
            cur_t_isotropic = compute_metrics(R, t, gtR, gtt)

            losses.append(loss.item())
            r_mse.append(cur_r_mse)
            r_mae.append(cur_r_mae)
            t_mse.append(cur_t_mse)
            t_mae.append(cur_t_mae)
            r_isotropic.append(cur_r_isotropic.cpu().detach().numpy())
            t_isotropic.append(cur_t_isotropic.cpu().detach().numpy())
    model.train()
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
        summary_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)
    results = {
        'loss': np.mean(losses),
        'r_mse': r_mse,
        'r_mae': r_mae,
        't_mse': t_mse,
        't_mae': t_mae,
        'r_isotropic': r_isotropic,
        't_isotropic': t_isotropic
    }
    return results


def main():
    args = config_params()
    print(args)

    setup_seed(args.seed)
    if not os.path.exists(args.saved_path):
        os.makedirs(args.saved_path)
    summary_path = os.path.join(args.saved_path, 'summary')
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    checkpoints_path = os.path.join(args.saved_path, 'checkpoints')
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    train_set = ModelNet40(root=args.root,
                           npts=args.train_npts,
                           train=True,
                           normal=args.normal,
                           mode=args.mode)
    test_set = ModelNet40(root=args.root,
                          npts=args.train_npts,
                          train=False,
                          normal=args.normal,
                          mode=args.mode)
    train_loader = DataLoader(train_set, batch_size=args.batchsize,
                              shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batchsize, shuffle=False,
                             num_workers=args.num_workers)

    #in_dim = 6 if args.normal else 3
    #model = IterativeBenchmark(in_dim=in_dim, niters=args.niters, gn=args.gn)
    model = RPMNet(args)
    model = model.cuda()
    loss_fn = EMDLosspy()
    loss_fn = loss_fn.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=args.milestones,
                                                     gamma=args.gamma,
                                                     last_epoch=-1)

    writer = SummaryWriter(summary_path)

    test_min_loss, test_min_r_mse_error, test_min_rot_error = \
        float('inf'), float('inf'), float('inf')
    for epoch in range(args.epoches):
        print('=' * 20, epoch + 1, '=' * 20)
        train_results = train_one_epoch(train_loader, model, loss_fn, optimizer)
        print_train_info(train_results)
        test_results = test_one_epoch(test_loader, model, loss_fn)
        print_train_info(test_results)

        if epoch % args.saved_frequency == 0:
            writer.add_scalar('Loss/train', train_results['loss'], epoch + 1)
            writer.add_scalar('Loss/test', test_results['loss'], epoch + 1)
            writer.add_scalar('RError/train', train_results['r_mse'], epoch + 1)
            writer.add_scalar('RError/test', test_results['r_mse'], epoch + 1)
            writer.add_scalar('rotError/train', train_results['r_isotropic'], epoch + 1)
            writer.add_scalar('rotError/test', test_results['r_isotropic'], epoch + 1)
            writer.add_scalar('Lr', optimizer.param_groups[0]['lr'], epoch + 1)
        test_loss, test_r_error, test_rot_error = \
            test_results['loss'], test_results['r_mse'], test_results['r_isotropic']
        if test_loss < test_min_loss:
            saved_path = os.path.join(checkpoints_path, "test_min_loss.pth")
            torch.save(model.state_dict(), saved_path)
            test_min_loss = test_loss
        if test_r_error < test_min_r_mse_error:
            saved_path = os.path.join(checkpoints_path, "test_min_rmse_error.pth")
            torch.save(model.state_dict(), saved_path)
            test_min_r_mse_error = test_r_error
        if test_rot_error < test_min_rot_error:
            saved_path = os.path.join(checkpoints_path, "test_min_rot_error.pth")
            torch.save(model.state_dict(), saved_path)
            test_min_rot_error = test_rot_error
        scheduler.step()


if __name__ == '__main__':
    main()