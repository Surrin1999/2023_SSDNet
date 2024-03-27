import torch.nn as nn
import torch
from model.model_util import *
from util import *
import matplotlib.pyplot as plt


class StereoSpixelNet(nn.Module):
    def __init__(self, num_iter, is_trainning=True, stereo=True):
        super(StereoSpixelNet, self).__init__()

        # define the element of networks
        self.feature_extractor = feature_network()
        self.pos_feature = position_encoding()
        self.pam = PAM(64)

        # define some parameters
        self.num_iter = num_iter
        self.is_training = is_trainning
        self.stereo = stereo

    def forward(self, l, r, left_img, right_img, p2sp_index, invisible, init_index,
                cir_index, problabel, spixel_h, spixel_w, device):
        # setting up some parameters
        self.num_spixels_h = spixel_h[0]
        self.num_spixels_w = spixel_w[0]
        self.num_spixels = spixel_h[0] * spixel_w[0]
        self.device = device

        # feature extractor
        left = left_img[:, 2:, :, :]
        right = right_img[:, 2:, :, :]
        left_feature = self.feature_extractor(left)
        right_feature = self.feature_extractor(right)

        # capture the correspondence between stereo images pair
        if self.is_training:
            stereo_feature, (M_right_to_left, M_left_to_right), \
            (M_left_right_left, M_right_left_right), \
            (V_left_to_right, V_right_to_left), wrap_features = self.pam(left_feature, right_feature, self.is_training)

        else:
            stereo_feature, wrap_features = self.pam(left_feature, right_feature, self.is_training)

        # adaptively fusion
        pos_feature = self.pos_feature(stereo_feature, left_img)

        # final features for clustering
        trans_features = torch.cat((pos_feature, left_img), 1)

        ############################################################## soft clustering module ###############################################################
        # compute superpixel center
        spixel_feature = SpixelFeature(trans_features, init_index, max_spixels=self.num_spixels)

        # soft clustering iteratively
        for i in range(self.num_iter):
            spixel_feature, _ = exec_iter(spixel_feature, trans_features, cir_index, p2sp_index, invisible,
                                          self.num_spixels_h, self.num_spixels_w, self.device)

        # get final Q map
        final_pixel_assoc = compute_assignments(spixel_feature, trans_features, p2sp_index, invisible, device)

        # get superpixel-level feature and get superpixel map
        new_spixel_feat = SpixelFeature2(left_img, final_pixel_assoc, cir_index, invisible, self.num_spixels_h,
                                         self.num_spixels_w)
        new_spix_indices = compute_final_spixel_labels(final_pixel_assoc, p2sp_index, self.num_spixels_h,
                                                       self.num_spixels_w)

        # reconstruct features(semantic label / spatial information / pixel)
        recon_feat2 = Semar(new_spixel_feat, new_spix_indices)
        spixel_label = SpixelFeature2(problabel, final_pixel_assoc, cir_index, invisible, self.num_spixels_h,
                                      self.num_spixels_w)
        recon_label = decode_features(final_pixel_assoc, spixel_label, p2sp_index, self.num_spixels_h,
                                      self.num_spixels_w, self.num_spixels, 50)
        ###################################################################################################################################################

        if self.is_training:
            if self.stereo:
                return recon_feat2, recon_label, new_spix_indices, (M_right_to_left, M_left_to_right), \
                       (M_left_right_left, M_right_left_right), \
                       (V_left_to_right, V_right_to_left)
            else:
                return recon_feat2, recon_label, new_spix_indices
        else:
            return recon_feat2, new_spix_indices

    def load_backbone(self, path):
        self.feature_extractor.load_state_dict(torch.load(path))

    def load_pam(self, path):
        self.pam.load_state_dict(torch.load(path))

    def set_num(self, sp_num):
        self.num_spixels = sp_num

    def set_iter(self, iter_num):
        self.num_iter = iter_num
