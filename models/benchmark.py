import open3d as o3d
import os
import torch
import torch.nn as nn
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOR_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOR_DIR)
from utils import batch_quat2mat, batch_transform


class PointNet(nn.Module):
    def __init__(self, in_dim=3, mlps=[64, 64, 64, 128, 1024]):
        super(PointNet, self).__init__()
        self.backbone = nn.Sequential()
        for i, out_dim in enumerate(mlps):
            self.backbone.add_module(f'pointnet_conv_{i}',
                                     nn.Conv1d(in_dim, out_dim, 1, 1, 0))
            #self.backbone.add_module(f'pointnet_bn_{i}',
            #                         nn.BatchNorm1d(out_dim))
            self.backbone.add_module(f'pointnet_relu_{i}',
                                     nn.ReLU(inplace=True))
            in_dim = out_dim

    def forward(self, x):
        x = self.backbone(x)
        x, _ = torch.max(x, dim=2)
        return x


class Benchmark(nn.Module):
    def __init__(self, in_dim1, in_dim2=2048, fcs=[1024, 1204, 512, 512, 256, 7]):
        super(Benchmark, self).__init__()
        self.encoder = PointNet(in_dim=in_dim1)
        self.decoder = nn.Sequential()
        for i, out_dim in enumerate(fcs):
            self.decoder.add_module(f'fc_{i}', nn.Linear(in_dim2, out_dim))
            if out_dim != 7:
                #self.decoder.add_module(f'bn_{i}', nn.BatchNorm1d(out_dim))
                self.decoder.add_module(f'relu_{i}', nn.ReLU(inplace=True))
            in_dim2 = out_dim

    def forward(self, x, y):
        x_f, y_f = self.encoder(x), self.encoder(y)
        concat = torch.cat((x_f, y_f), dim=1)
        out = self.decoder(concat)
        batch_t, batch_quat = out[:, :3], out[:, 3:] / torch.norm(out[:, 3:], dim=1, keepdim=True)
        batch_R = batch_quat2mat(batch_quat)
        transformed_x = batch_transform(x.permute(0, 2, 1).contiguous(),
                                        batch_R, batch_t)
        return batch_R, batch_t, transformed_x


class IterativeBenchmark(nn.Module):
    def __init__(self, in_dim1, niters):
        super(IterativeBenchmark, self).__init__()
        self.benckmark = Benchmark(in_dim1)
        self.niters = niters

    def forward(self, x, y):
        device = x.device
        B = x.size()[0]
        transformed_x = torch.clone(x)
        batch_R_res = torch.eye(3).to(device).unsqueeze(0).repeat(B, 1, 1)
        batch_t_res = torch.zeros(3, 1).to(device).unsqueeze(0).repeat(B, 1, 1)
        for i in range(self.niters):
            batch_R, batch_t, transformed_x = self.benckmark(transformed_x, y)
            batch_R_res = torch.matmul(batch_R, batch_R_res)
            batch_t_res = torch.matmul(batch_R, batch_t_res) \
                          + torch.unsqueeze(batch_t, -1)
            transformed_x = transformed_x.permute(0, 2, 1).contiguous()
        batch_t_res = torch.squeeze(batch_t_res)
        transformed_x = transformed_x.permute(0, 2, 1).contiguous()
        return batch_R_res, batch_t_res, transformed_x


if __name__ == '__main__':
    x, y = torch.randn(4, 3, 5), torch.randn(4, 3, 5)
    net = IterativeBenchmark(in_dim1=3, niters=2)
    print(net)
    batch_R, batch_t, transformed_x = net(x, y)
    print(batch_R, batch_R.size())
    print(batch_t, batch_t.size())
