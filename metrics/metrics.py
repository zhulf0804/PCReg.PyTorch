import math
import torch


def translation_error(t1, t2):
    '''
    calculate translation error.
    :param t1: shape=(B, 3)
    :param t2: shape=(B, 3)
    :return:
    '''
    error = torch.mean(torch.sum((t1 - t2)**2, dim=1))
    return error


def degree_error(r1, r2):
    '''
    Calculate degree error between r1 and r2.
    :param r1: shape=(B, 3, 3), pred
    :param r2: shape=(B, 3, 3), gt
    :return:
    '''
    r2_inv = r2.permute(0, 2, 1).contiguous()
    device = r1.device
    B = r1.shape[0]
    r1r2 = torch.matmul(r2_inv, r1)
    mask = torch.unsqueeze(torch.eye(3).to(device), dim=0).repeat(B, 1, 1)
    tr = torch.sum(torch.reshape(mask * r1r2, (B, 9)), dim=-1)
    rads = torch.acos(torch.clamp((tr - 1) / 2, -1, 1))
    degrees = torch.abs(rads / math.pi * 180)
    degree = torch.mean(degrees)
    return degree


def rotation_error(r1, r2):
    '''
    calculate mse r1-r2 error.
    :param r1: shape=(B, 3, 3), pred
    :param r2: shape=(B, 3, 3), gt
    :return:
    '''
    r = torch.reshape(r1 - r2, (-1, 9))
    error = torch.mean(torch.sum(r ** 2, dim=1))
    return error


def angle_error(r1, r2):
    '''

    :param r1: shape=(B, 3, 3), pred
    :param r2: shape=(B, 3, 3), gt
    :return:
    '''


if __name__ == '__main__':
    r1 = torch.randn(2, 3, 3)
    r2 = torch.randn(2, 3, 3)
    r2[0] = torch.inverse(r1[0])
    r2[1] = torch.inverse(r1[1])
    r_error = rotation_error(r1, r2)
    print(r_error)
    r_error2 = r_error.item()
    print(r_error2)