import torch
import torch.nn as nn
import numpy as np
from util import *
from loss import *
from skimage import morphology
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from skimage.color import lab2rgb


################################# feature extraction #################################
# basic block for our network
class conv_bn_relu(nn.Module):
    def __init__(self, in_channels, channels, bn=True):
        super(conv_bn_relu, self).__init__()
        self.BN_ = bn
        self.conv = nn.Conv2d(in_channels, channels, 3, padding=1)
        if self.BN_:
            self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.BN_:
            x = self.bn(x)
        x = self.relu(x)
        return x


# ASPP for more context information
class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net


# feature extractor
class feature_network(nn.Module):
    def __init__(self):
        super(feature_network, self).__init__()
        self.conv1 = conv_bn_relu(3, 64)
        self.conv2 = conv_bn_relu(64, 64)

        self.pool1 = nn.MaxPool2d(3, 2, 1)
        self.conv3 = conv_bn_relu(64, 64)
        self.conv4 = conv_bn_relu(64, 64)

        self.pool2 = nn.MaxPool2d(3, 2, 1)
        self.conv5 = conv_bn_relu(64, 64)
        self.conv6 = conv_bn_relu(64, 64)
        self.conv7 = conv_bn_relu(64 * 3, 64)

        # self.aspp = ASPP(in_channel=64, depth=64)

        self.conv4_up = nn.Upsample(scale_factor=2)
        self.conv6_up = nn.Upsample(scale_factor=4)

    def forward(self, x):
        conv2 = self.conv2(self.conv1(x))
        conv4 = self.conv4(self.conv3(self.pool1(conv2)))
        conv6 = self.conv6(self.conv5(self.pool2(conv4)))
        # aspp = self.aspp(conv6)

        _, _, h, w = x.shape
        out = self.conv7(
            torch.cat((conv2[:, :, :h, :w], self.conv4_up(conv4)[:, :, :h, :w], self.conv6_up(conv6)[:, :, :h, :w]),
                      dim=1))
        return out


# # ssn网络中的卷积网络部分
# class cnn_module(nn.Module):
#     def __init__(self, in_channel=5, out_channel=15):
#         super(cnn_module, self).__init__()
#         self.conv1 = conv_bn_relu(in_channel, 64)
#         self.conv2 = conv_bn_relu(64, 64)
#         self.pool1 = nn.MaxPool2d(3, 2, 1)
#
#         self.conv3 = conv_bn_relu(64, 64)
#         self.conv4 = conv_bn_relu(64, 64)
#         self.pool2 = nn.MaxPool2d(3, 2, 1)
#
#         self.conv5 = conv_bn_relu(64, 64)
#         self.conv6 = conv_bn_relu(64, 64)
#
#         self.conv6_up = nn.Upsample(scale_factor=4)
#         self.conv4_up = nn.Upsample(scale_factor=2)
#
#         self.conv7 = conv_bn_relu(64 * 3 + in_channel, out_channel, False)
#
#     def forward(self, x):
#         conv1 = self.conv1(x)
#         conv2 = self.conv2(conv1)
#         pool1 = self.pool1(conv2)
#
#         conv3 = self.conv3(pool1)
#         conv4 = self.conv4(conv3)
#         pool2 = self.pool2(conv4)
#
#         conv5 = self.conv5(pool2)
#         conv6 = self.conv6(conv5)
#
#         conv6_up = self.conv6_up(conv6)
#         conv4_up = self.conv4_up(conv4)
#
#         _, _, h, w = x.shape
#         conv_concat = torch.cat((x, conv2[:, :, :h, :w], conv4_up[:, :, :h, :w], conv6_up[:, :, :h, :w]), 1)
#         conv7 = self.conv7(conv_concat)
#
#         return conv7

#######################################################################################


############################### stereo alignment module ###############################
class PAM(nn.Module):
    def __init__(self, channels):
        super(PAM, self).__init__()
        self.b1 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b2 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b3 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.softmax = nn.Softmax(-1)
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def __call__(self, x_left, x_right, is_training=True):
        b, c, h, w = x_left.shape

        buffer_left = x_left
        buffer_right = x_right

        ### M_{right_to_left}
        Q = self.b1(buffer_left).permute(0, 2, 3, 1)  # B * H * W * C
        S = self.b2(buffer_right).permute(0, 2, 1, 3)  # B * H * C * W

        score = torch.bmm(Q.contiguous().view(-1, w, c),
                          S.contiguous().view(-1, c, w))  # (B*H) * W * W
        M_right_to_left = self.softmax(score)

        # map = M_right_to_left[50].cpu().detach().numpy()
        # plt.imshow(map)
        # plt.show()
        ### M_{left_to_right}
        Q = self.b1(buffer_right).permute(0, 2, 3, 1)  # B * H * W * C
        S = self.b2(buffer_left).permute(0, 2, 1, 3)  # B * H * C * W
        score = torch.bmm(Q.contiguous().view(-1, w, c),
                          S.contiguous().view(-1, c, w))  # (B*H) * W * W
        M_left_to_right = self.softmax(score)

        ### valid masks
        V_left_to_right = torch.sum(M_left_to_right.detach(), 1) > 0.1
        V_left_to_right = V_left_to_right.view(b, 1, h, w)  # B * 1 * H * W
        V_left_to_right = morphologic_process(V_left_to_right)
        if is_training == 1:
            V_right_to_left = torch.sum(M_right_to_left.detach(), 1) > 0.1
            V_right_to_left = V_right_to_left.view(b, 1, h, w)  # B * 1 * H * W
            V_right_to_left = morphologic_process(V_right_to_left)

            M_left_right_left = torch.bmm(M_right_to_left, M_left_to_right)
            M_right_left_right = torch.bmm(M_left_to_right, M_right_to_left)

        ### fusion
        buffer = self.b3(x_right).permute(0, 2, 3, 1).contiguous().view(-1, w, c)  # (B*H) * W * C
        buffer = torch.bmm(M_right_to_left, buffer).contiguous().view(b, h, w, c).permute(0, 3, 1, 2)  # B * C * H * W
        buffer = torch.mul(buffer, V_left_to_right)
        wrap_features = buffer + torch.mul(x_left, 1 - V_left_to_right)
        out = self.fusion(torch.cat((wrap_features, x_left), 1))

        ## output
        if is_training == 1:
            return out, \
                   (M_right_to_left.contiguous().view(b, h, w, w), M_left_to_right.contiguous().view(b, h, w, w)), \
                   (M_left_right_left.view(b, h, w, w), M_right_left_right.view(b, h, w, w)), \
                   (V_left_to_right, V_right_to_left), wrap_features
        if is_training == 0:
            return out, buffer


def morphologic_process(mask):
    device = mask.device
    b, _, _, _ = mask.shape
    mask = ~mask
    mask_np = mask.cpu().numpy().astype(bool)
    mask_np = morphology.remove_small_objects(mask_np, 20, 2)
    mask_np = morphology.remove_small_holes(mask_np, 10, 2)
    for idx in range(b):
        buffer = np.pad(mask_np[idx, 0, :, :], ((3, 3), (3, 3)), 'constant')
        buffer = morphology.binary_closing(buffer, morphology.disk(3))
        mask_np[idx, 0, :, :] = buffer[3:-3, 3:-3]
    mask_np = 1 - mask_np
    mask_np = mask_np.astype(float)

    return torch.from_numpy(mask_np).float().to(device)


#######################################################################################


################################## position encoding ##################################
class position_encoding(nn.Module):
    def __init__(self):
        super(position_encoding, self).__init__()
        self.reduce_dim_x = nn.Conv2d(64, 16, 1)
        self.reduce_dim_y = nn.Conv2d(64, 16, 1)
        self.conv1 = conv_bn_relu(96, 64)
        self.adp_fusion = channel_fusion(64)
        self.conv2 = conv_bn_relu(64, 15)

    def forward(self, features, img):
        # b, _, h, w = img.shape
        x_position = img[:, 0:1, :, :] / torch.max(img[:, 0:1, :, :])
        y_position = img[:, 1:2, :, :] / torch.max(img[:, 1:2, :, :])

        x_feature = self.reduce_dim_x((features + x_position))
        y_feature = self.reduce_dim_y((features + y_position))

        concat_feature = torch.cat((x_feature, y_feature, features), 1)
        out = self.conv2(self.adp_fusion(self.conv1(concat_feature)))

        return out


#######################################################################################


############################### adaptively fusion module ##############################
class channel_fusion(nn.Module):
    def __init__(self, channels, r=16):
        super(channel_fusion, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.squeeze = nn.Sequential(
            nn.Conv2d(channels, channels // r, 1),
            nn.LayerNorm([channels // r, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // r, channels, 1),
            nn.Sigmoid(),
        )
        self.conv1 = conv_bn_relu(channels, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        x1 = self.conv1(x)
        y = self.avg_pool(x).view(b, c, 1, 1)
        y = self.squeeze(y).view(b, c, 1, 1)
        y = torch.mul(x1, y)
        y = x + y

        return y

#######################################################################################
