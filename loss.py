import torch
import torch.nn as nn
import numpy as np
import pdb
from torch.autograd import Variable
import matplotlib.pyplot as plt


class position_color_loss(nn.Module):
    def __init__(self, pos_weight=0, col_weight=0):
        """
        :param pos_weight:
        :param col_weight:
        """
        super(position_color_loss, self).__init__()
        self.pos_weight = pos_weight
        self.col_weight = col_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, recon_feat, pixel_features):
        """

        :param recon_feat: B*C*H*W restructure  pixel feature (c=RGBplusXY)
        :param pixel_features: B*C*H*W original pixel feature
        :return:
        """
        # pdb.set_trace()
        pos_recon_feat = recon_feat[:, :2, :, :]
        color_recon_feat = recon_feat[:, 2:, :, :]
        pos_pix_feat = pixel_features[:, :2, :, :]
        color_pix_feat = pixel_features[:, 2:, :, :]

        pos_loss = self.mse_loss(pos_recon_feat, pos_pix_feat)
        color_loss = self.mse_loss(color_recon_feat, color_pix_feat)

        pos_clor_loss = pos_loss * self.pos_weight + color_loss * self.col_weight

        return pos_clor_loss


class LossWithoutSoftmax(nn.Module):
    def __init__(self, loss_weight=1.0, ignore_label=255):
        super(LossWithoutSoftmax, self).__init__()
        self.loss_weight = loss_weight
        self.ignore_label = ignore_label
        self.NLLloss = nn.NLLLoss(reduction='none')

    def forward(self, recon_label3, label, invisible_p=None):
        """
        :param recon_label3: B*C*H*W  reconstructure label by soft threshold
        :param label:  B*1*H*W gt label
        :param invisible_p: B*H*W invisible pixel (ignore region)
        :return:
        """
        # pdb.set_trace()
        label = label[:, 0, ...]

        # add ignore region
        if invisible_p is not None:
            ignore = invisible_p == 1.
        elif self.ignore_label is not None:
            ignore = label == self.ignore_label
        else:
            raise IOError
        label[ignore] = 0

        loss = self.NLLloss(recon_label3, label)  # B*H*W
        #
        # view_loss = loss.data.numpy()
        #
        loss = -1 * loss[~ignore]
        loss = -1 * torch.log(loss)
        loss = loss.mean() * self.loss_weight

        return loss


def smooth_loss(M_right_to_left, M_left_to_right):
    criterion_L1 = nn.L1Loss().cuda()
    loss_h = criterion_L1(M_right_to_left[:, :-1, :, :], M_right_to_left[:, 1:, :, :]) + \
             criterion_L1(M_left_to_right[:, :-1, :, :], M_left_to_right[:, 1:, :, :])
    loss_w = criterion_L1(M_right_to_left[:, :, :-1, :-1], M_right_to_left[:, :, 1:, 1:]) + \
             criterion_L1(M_left_to_right[:, :, :-1, :-1], M_left_to_right[:, :, 1:, 1:])
    loss_smooth = loss_w + loss_h

    return loss_smooth


def cycle_loss(V_right_to_left, V_left_to_right, M_left_right_left, M_right_left_right, w, b, h):
    criterion_L1 = nn.L1Loss().cuda()
    Identity = Variable(torch.eye(w, w).repeat(b, h, 1, 1), requires_grad=False).cuda()
    loss_cycle = criterion_L1(M_left_right_left * V_left_to_right.permute(0, 2, 1, 3),
                              Identity * V_left_to_right.permute(0, 2, 1, 3)) + \
                 criterion_L1(M_right_left_right * V_right_to_left.permute(0, 2, 1, 3),
                              Identity * V_right_to_left.permute(0, 2, 1, 3))
    return loss_cycle


def photometric_loss(M_right_to_left, M_left_to_right, V_left_to_right, V_right_to_left,
                     image_left, image_right, b, h, w, c):
    criterion_L1 = nn.L1Loss().cuda()
    # print(M_right_to_left.shape)
    LR_right_warped = torch.bmm(M_right_to_left.contiguous().view(b * h, w, w),
                                image_right.permute(0, 2, 3, 1).contiguous().view(b * h, w, c))
    LR_right_warped = LR_right_warped.view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
    LR_left_warped = torch.bmm(M_left_to_right.contiguous().view(b * h, w, w),
                               image_left.permute(0, 2, 3, 1).contiguous().view(b * h, w, c))
    LR_left_warped = LR_left_warped.view(b, h, w, c).contiguous().permute(0, 3, 1, 2)

    loss_photo = criterion_L1(image_left * V_left_to_right, LR_right_warped * V_left_to_right) + \
                 criterion_L1(image_right * V_right_to_left, LR_left_warped * V_right_to_left)
    # l = LR_right_warped[0].permute(1,2,0).cpu().detach().numpy()
    # plt.imshow(l)
    # plt.show()
    return loss_photo


def zero_occlusion(V_left_to_right, V_right_to_left):
    L = nn.L1Loss().cuda()
    zero_occ = torch.zeros_like(V_left_to_right).cuda()

    loss = torch.exp(- L(V_left_to_right, zero_occ) - L(V_right_to_left, zero_occ))
    return loss


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.loss1 = position_color_loss()
        self.loss2 = LossWithoutSoftmax()

    def forward(self, recon_feat2, pixel_feature, recon_label, label):
        loss1 = self.loss1(recon_feat2, pixel_feature)
        loss2 = self.loss2(recon_label, label)

        return loss2 + loss1, loss1, loss2
