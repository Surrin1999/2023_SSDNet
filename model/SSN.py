from .model_util import conv_bn_relu
import torch
import torch.nn as nn
from util import *


class cnn_module(nn.Module):
    def __init__(self, out_channel=15):
        super(cnn_module, self).__init__()
        self.conv1 = conv_bn_relu(5, 64)
        self.conv2 = conv_bn_relu(64, 64)
        self.pool1 = nn.MaxPool2d(3, 2, 1)

        self.conv3 = conv_bn_relu(64, 64)
        self.conv4 = conv_bn_relu(64, 64)
        self.pool2 = nn.MaxPool2d(3, 2, 1)

        self.conv5 = conv_bn_relu(64, 64)
        self.conv6 = conv_bn_relu(64, 64)

        self.conv6_up = nn.Upsample(scale_factor=4)
        self.conv4_up = nn.Upsample(scale_factor=2)

        self.conv7 = conv_bn_relu(197, out_channel, False)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        pool1 = self.pool1(conv2)

        conv3 = self.conv3(pool1)
        conv4 = self.conv4(conv3)
        pool2 = self.pool2(conv4)

        conv5 = self.conv5(pool2)
        conv6 = self.conv6(conv5)

        conv6_up = self.conv6_up(conv6)
        conv4_up = self.conv4_up(conv4)

        _, _, h, w = x.shape
        conv_concat = torch.cat((x, conv2[:, :, :h, :w], conv4_up[:, :, :h, :w], conv6_up[:, :, :h, :w]), 1)
        conv7 = self.conv7(conv_concat)
        conv_comb = torch.cat((x, conv7), 1)

        return conv_comb


class create_ssn_net(nn.Module):
    def __init__(self, num_spixels, num_iter, num_spixels_h, num_spixels_w, dtype='train', ssn=1):
        super(create_ssn_net, self).__init__()
        self.trans_features = cnn_module()
        self.num_spixels = num_spixels
        self.num_iter = num_iter
        self.num_spixels_h = num_spixels_h
        self.num_spixels_w = num_spixels_w
        self.num_spixels = num_spixels_h * num_spixels_w
        self.dtype = dtype
        self.ssn = ssn

    def forward(self, x, p2sp_index, invisible, init_index, cir_index, problabel, spixel_h, spixel_w, device):
        if self.ssn:
            trans_features = self.trans_features(x)
        else:
            trans_features = x
        self.num_spixels_h = spixel_h[0]
        self.num_spixels_w = spixel_w[0]
        self.num_spixels = spixel_h[0] * spixel_w[0]
        self.device = device

        # init spixel feature
        spixel_feature = SpixelFeature(trans_features, init_index, max_spixels=self.num_spixels)

        for i in range(self.num_iter):
            spixel_feature, _ = exec_iter(spixel_feature, trans_features, cir_index, p2sp_index,
                                          invisible, self.num_spixels_h, self.num_spixels_w, self.device)

        final_pixel_assoc = compute_assignments(spixel_feature, trans_features, p2sp_index, invisible,
                                                device)  # out of memory

        if self.dtype == 'train':
            new_spixel_feat = SpixelFeature2(x, final_pixel_assoc, cir_index, invisible,
                                             self.num_spixels_h, self.num_spixels_w)
            new_spix_indices = compute_final_spixel_labels(final_pixel_assoc, p2sp_index,
                                                           self.num_spixels_h, self.num_spixels_w)
            recon_feat2 = Semar(new_spixel_feat, new_spix_indices)
            spixel_label = SpixelFeature2(problabel, final_pixel_assoc, cir_index, invisible,
                                          self.num_spixels_h, self.num_spixels_w)
            recon_label = decode_features(final_pixel_assoc, spixel_label, p2sp_index,
                                          self.num_spixels_h, self.num_spixels_w, self.num_spixels, 50)
            return recon_feat2, recon_label

        elif self.dtype == 'test':
            new_spixel_feat = SpixelFeature2(x, final_pixel_assoc, cir_index, invisible,
                                             self.num_spixels_h, self.num_spixels_w)
            new_spix_indices = compute_final_spixel_labels(final_pixel_assoc, p2sp_index,
                                                           self.num_spixels_h, self.num_spixels_w)
            recon_feat2 = Semar(new_spixel_feat, new_spix_indices)
            spixel_label = SpixelFeature2(problabel, final_pixel_assoc, cir_index, invisible,
                                          self.num_spixels_h, self.num_spixels_w)
            recon_label = decode_features(final_pixel_assoc, spixel_label, p2sp_index,
                                          self.num_spixels_h, self.num_spixels_w, self.num_spixels, 50)

            # import pdb
            # pdb.set_trace()
            return recon_feat2, recon_label, new_spix_indices

        else:
            pass

# if __name__ == '__main__':
#     model = create_ssn_net(num_spixels=400, num_iter=10,
#                            num_spixels_h=10, num_spixels_w=10, dtype='test',
#                            ssn=0)
#     model = torch.nn.DataParallel(model)
#     model.load_state_dict(torch.load('../45000_0.527_model.pt'))
#
#     print(model)
