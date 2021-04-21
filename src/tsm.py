# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalShift(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_segment=3, n_div=2, inplace=False):
        super(TemporalShift, self).__init__()
        self.net = nn.Conv2d(in_channels, out_channels, kernel_size, padding=[2,2])
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        shape = x.size()
        tmp_feature = self.net(x.view(shape[0]*shape[1], shape[2], shape[3], shape[4] ))
        shape_1 = tmp_feature.shape
        return tmp_feature.view(shape_1[0]//2, 2, shape_1[1], shape_1[2], shape_1[3])

    @staticmethod
    def shift(x, n_segment, fold_div=2, inplace=False):
        batch, t, c, h, w = x.size()
        fold = c // fold_div
        if inplace:
            # Due to some out of order error when performing parallel computing.
            # May need to write a CUDA kernel.
            raise NotImplementedError
            # out = InplaceShift.apply(x, fold)
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            # out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, fold:] = x[:, :, fold:]  # not shift

        return out.view(batch, t, c, h, w)