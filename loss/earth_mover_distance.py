import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


def emd(template: torch.Tensor, source: torch.Tensor):
	from emd import EMDLoss
	emd_loss = torch.mean(EMDLoss()(template, source))/(template.size()[1])
	return emd_loss


class EMDLosspy(nn.Module):
	def __init__(self):
		super(EMDLosspy, self).__init__()

	def forward(self, template, source):
		return emd(template, source)


if __name__ == '__main__':
    loss = EMDLosspy()
    a = torch.randn(4, 5, 3).cuda()
    b = copy.deepcopy(a)
    v = loss(a, b)
    print(v)
