from abc import ABC

from torch import nn
from torch.nn import functional as F

from citizenuav.math import gram_matrix


class TensorBasedLoss(nn.Module, ABC):
    target = None

    def to(self, device):
        self.target = self.target.to(device)
        return super().to(device)
    
    def cuda(self, device=None):
        self.target = self.target.cuda()
        return super().cuda(device)

    def cpu(self):
        self.target = self.target.cpu()
        return super().cpu()

    def double(self):
        self.target = self.target.double()
        return super().double()

    def float(self):
        self.target = self.target.float()
        return super().float()


class StyleLoss(TensorBasedLoss):
    """
    Source: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
    """

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = None

    def forward(self, x):
        G = gram_matrix(x)
        self.loss = F.mse_loss(G, self.target)
        return x


class ContentLoss(TensorBasedLoss):
    """
    Source: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
    """

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise, the forward method of the criterion
        # will throw an error.
        self.target = target.detach()
        self.loss = None

    def forward(self, x):
        self.loss = F.mse_loss(x, self.target)
        return x
