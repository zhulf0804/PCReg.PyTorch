import argparse
import numpy as np
import open3d
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import ModelNet40
from models import IterativeBenchmark
from loss import EMDLosspy
from metrics import rotation_error, translation_error, degree_error
from utils import time_calc


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
    ## models training
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--epoches', type=int, default=400)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--in_dim', type=int, default=3,
                        help='3 for (x, y, z) or 6 for (x, y, z, nx, ny, nz)')
    parser.add_argument('--niters', type=int, default=8,
                        help='iteration nums in one model forward')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate')
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


@time_calc
def train_one_epoch(train_loader, model, loss_fn, optimizer):
    losses, t_errors, R_errors, degree_errors = [], [], [], []
    for ref_cloud, src_cloud, gtR, gtt in tqdm(train_loader):
        ref_cloud, src_cloud, gtR, gtt = ref_cloud.cuda(), src_cloud.cuda(), \
                                         gtR.cuda(), gtt.cuda()
        optimizer.zero_grad()
        R, t, pred_ref_cloud = model(src_cloud.permute(0, 2, 1).contiguous(),
                                     ref_cloud.permute(0, 2, 1).contiguous())
        loss = loss_fn(ref_cloud, pred_ref_cloud)
        loss.backward()
        optimizer.step()

        cur_t_error = translation_error(t, -gtt)
        cur_R_error = rotation_error(R, gtR.permute(0, 2, 1).contiguous())
        cur_degree_error = degree_error(R, gtR.permute(0, 2, 1).contiguous())
        losses.append(loss.item())
        t_errors.append(cur_t_error.item())
        R_errors.append(cur_R_error.item())
        degree_errors.append(cur_degree_error.item())
    return np.mean(losses), np.mean(t_errors), np.mean(R_errors), np.mean(degree_errors)


@time_calc
def test_one_epoch(test_loader, model, loss_fn):
    model.eval()
    losses, t_errors, R_errors, degree_errors = [], [], [], []
    with torch.no_grad():
        for ref_cloud, src_cloud, gtR, gtt in tqdm(test_loader):
            ref_cloud, src_cloud, gtR, gtt = ref_cloud.cuda(), src_cloud.cuda(), \
                                             gtR.cuda(), gtt.cuda()
            R, t, pred_ref_cloud = model(src_cloud.permute(0, 2, 1).contiguous(),
                                         ref_cloud.permute(0, 2, 1).contiguous())
            loss = loss_fn(ref_cloud, pred_ref_cloud)
            cur_t_error = translation_error(t, -gtt)
            cur_R_error = rotation_error(R, gtR.permute(0, 2, 1).contiguous())
            cur_degree_error = degree_error(R,
                                            gtR.permute(0, 2, 1).contiguous())
            losses.append(loss.item())
            t_errors.append(cur_t_error.item())
            R_errors.append(cur_R_error.item())
            degree_errors.append(cur_degree_error.item())
    model.train()
    return np.mean(losses), np.mean(t_errors), np.mean(R_errors), np.mean(
            degree_errors)


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

    train_set = ModelNet40(args.root, args.train_npts)
    test_set = ModelNet40(args.root, args.train_npts, False)
    train_loader = DataLoader(train_set, batch_size=args.batchsize,
                              shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batchsize, shuffle=False,
                             num_workers=args.num_workers)

    model = IterativeBenchmark(in_dim1=args.in_dim, niters=args.niters)
    model = model.cuda()
    loss_fn = EMDLosspy()
    loss_fn = loss_fn.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=args.milestones,
                                                     gamma=args.gamma,
                                                     last_epoch=-1)

    writer = SummaryWriter(summary_path)

    test_min_loss, test_min_t_error, test_min_R_error, test_min_degree_error = \
        float('inf'), float('inf'), float('inf'), float('inf')
    for epoch in range(args.epoches):
        print('=' * 20, epoch + 1, '=' * 20)
        train_loss, train_t_error, train_R_error, train_degree_error = \
            train_one_epoch(train_loader, model, loss_fn, optimizer)
        print('Train: loss: {:.4f}, t_error: {:.4f}, R_error: {:.4f}, degree_error: {:.4f}'.
              format(train_loss, train_t_error, train_R_error, train_degree_error))
        test_loss, test_t_error, test_R_error, test_degree_error = \
            test_one_epoch(test_loader, model, loss_fn)
        print('Test: loss: {:.4f}, t_error: {:.4f}, R_error: {:.4f}, degree_error: {:.4f}'.
              format(test_loss, test_t_error, test_R_error, test_degree_error))

        if epoch % args.saved_frequency == 0:
            writer.add_scalar('Loss/train', train_loss, epoch + 1)
            writer.add_scalar('Loss/test', test_loss, epoch + 1)
            writer.add_scalar('TError/train', train_t_error, epoch + 1)
            writer.add_scalar('TError/test', test_t_error, epoch + 1)
            writer.add_scalar('RError/train', train_R_error, epoch + 1)
            writer.add_scalar('RError/test', test_R_error, epoch + 1)
            writer.add_scalar('degreeError/train', train_degree_error, epoch + 1)
            writer.add_scalar('degreeError/test', test_degree_error, epoch + 1)
            writer.add_scalar('Lr', optimizer.param_groups[0]['lr'], epoch + 1)
        if test_loss < test_min_loss:
            saved_path = os.path.join(checkpoints_path, "test_min_loss.pth")
            torch.save(model.state_dict(), saved_path)
            test_min_loss = test_loss
        if test_t_error < test_min_t_error:
            saved_path = os.path.join(checkpoints_path, "test_min_t_error.pth")
            torch.save(model.state_dict(), saved_path)
            test_min_t_error = test_t_error
        if test_R_error < test_min_R_error:
            saved_path = os.path.join(checkpoints_path, "test_min_R_error.pth")
            torch.save(model.state_dict(), saved_path)
            test_min_R_error = test_R_error
        if test_degree_error < test_min_degree_error:
            saved_path = os.path.join(checkpoints_path,
                                     "test_min_degree_error.pth")
            torch.save(model.state_dict(), saved_path)
            test_min_degree_error = test_degree_error
        scheduler.step()


if __name__ == '__main__':
    main()