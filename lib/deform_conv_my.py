# This transform the tensorflow version of https://github.com/maestrojeong/deformable_convnet
# But there is something wrong in my code.

import torch
from torch import nn
from torch.autograd import Variable
# from config import *
import numpy as np


class DeformConv(nn.Module):
    def __init__(self, offset_shape, filter_shape, config=None):
        super(DeformConv, self).__init__()
        self.f_h, self.f_w, self.f_ic, self.f_oc = filter_shape
        self.o_h, self.o_w, self.o_ic, self.o_oc = offset_shape

        if not config:
            self.cuda_num = 0
        else:
            self.cuda_num = config.cuda_num

        assert self.o_oc == 2 * self.f_h * self.f_w

        self.offset_conv = nn.Sequential(
            nn.Conv2d(self.o_ic, self.o_oc, (self.o_h, self.o_w), stride=1),
            # nn.ReLU(),
            # nn.BatchNorm2d(self.o_oc),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(self.f_ic, self.f_oc, kernel_size=(self.f_h, self.f_w), stride=(self.f_h, self.f_w))
        )

    def forward(self, x):
        b, i_c, i_h, i_w,  = x.size()
        assert self.f_ic == i_c and self.o_ic == i_c

        width_left = (self.o_w - 1) // 2
        width_right = (self.o_w - 1) - width_left

        height_top = (self.o_h - 1) // 2
        height_bottom = (self.o_h - 1) - height_top

        pad = [width_left, width_right, height_top, height_bottom]
        x_offset = nn.functional.pad(x, pad)
        # (batch, 2*fh*fw, ih, iw)
        offset_map = self.offset_conv(x_offset)

        # (batch, ih, iw,  2*fh*fw)
        offset_map = offset_map.permute(0, 2, 3, 1).contiguous()
        offset_map = offset_map.view(b, i_h, i_w, self.f_h, self.f_w, 2)

        offset_map_h = offset_map[..., 0].contiguous()
        offset_map_h = offset_map_h.view(b, i_h, i_w, self.f_h, self.f_w)

        offset_map_w = offset_map[..., 1].contiguous()
        offset_map_w = offset_map_w.view(b, i_h, i_w, self.f_h, self.f_w)

        coord_h, coord_w = np.meshgrid(range(i_h), range(i_w), indexing='ij')
        coord_w = Variable(torch.from_numpy(coord_w).type(torch.FloatTensor))
        coord_h = Variable(torch.from_numpy(coord_h).type(torch.FloatTensor))

        coord_fh, coord_fw = np.meshgrid(range(self.f_w), range(self.f_h), indexing='ij')
        coord_fw = Variable(torch.from_numpy(coord_fw).type(torch.FloatTensor))
        coord_fh = Variable(torch.from_numpy(coord_fh).type(torch.FloatTensor))

        if torch.cuda.is_available():
            coord_h = coord_h.cuda(self.cuda_num)
            coord_w = coord_w.cuda(self.cuda_num)
            coord_fw = coord_fw.cuda(self.cuda_num)
            coord_fh = coord_fh.cuda(self.cuda_num)

        coord_h = coord_h.view(1, i_h, i_w, 1, 1).repeat(b, 1, 1, self.f_h, self.f_w)
        coord_w = coord_w.view(1, i_h, i_w, 1, 1).repeat(b, 1, 1, self.f_h, self.f_w)

        coord_fh = coord_fh.view(1, 1, 1, self.f_h, self.f_w).repeat(b, i_h, i_w, 1, 1)
        coord_fw = coord_fw.view(1, 1, 1, self.f_h, self.f_w).repeat(b, i_h, i_w, 1, 1)

        coord_h = coord_h + coord_fh
        coord_w = coord_w + coord_fw

        # coord_h = coord_h + coord_fh + offset_map_h
        # coord_w = coord_w + coord_fw + offset_map_w

        coord_h = torch.clamp(coord_h, min=0, max=i_h - 1)
        coord_w = torch.clamp(coord_w, min=0, max=i_w - 1)

        # points_grid = torch.cat([coord_h.unsqueeze(-1), coord_w.unsqueeze(-1)], dim=-1).view(-1, self.f_h, self.f_w, 2).contiguous()
        # x = x.repeat(i_h * i_w, 1, 1, 1).contiguous()
        # sample_feature = torch.nn.functional.grid_sample(x, points_grid).view(b, i_h, i_w, i_c, self.f_h, self.f_w)
        # sample_feature = sample_feature.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, i_c, i_h * self.f_h, i_w * self.f_w)

        points_grid = torch.cat([coord_h.unsqueeze(-1), coord_w.unsqueeze(-1)], dim=-1)
        points_grid = points_grid.permute(1, 2, 0, 3, 4, 5).contiguous().view(-1, b, self.f_h, self.f_w, 2)
        points_sample = []
        for point_index in range(points_grid.size(0)):
            # print(x.size(), points_grid.size())
            sample = torch.nn.functional.grid_sample(x, points_grid[point_index], mode='bilinear').unsqueeze(0)
            points_sample.append(sample)
        points_sample = torch.cat(points_sample, 0)

        points_sample = points_sample.view(i_h, i_w, b, i_c, self.f_h, self.f_w).permute(2, 3, 0, 4, 1, 5).contiguous()
        sample_feature = points_sample.view(b, i_c, i_h * self.f_h, i_w * self.f_w)

        res = self.conv(sample_feature)

        # print(res.size())
        # print(res[0,0, :3, :3])
        return res

# if __name__ == '__main__':
    # t = time.time()
    # # (1, 10, 10, 3)
    # x = Variable(torch.from_numpy(img))
    # x = x.permute(0, 3, 1, 2)
    #
    # # offset_map = Variable(torch.from_numpy(offset_m))
    # # (3, 3, 3, 2 * 4, 4)
    # offset_shape = [oh, ow, oic, ooc]
    # # (4, 4, 3, 5)
    # filter_shape = [fh, fw, fic, foc]
    #
    # obj = DeformConv(offset_shape, filter_shape)
    # obj.forward(x)
    #
    # print(time.time() - t)
