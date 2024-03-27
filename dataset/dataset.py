from torch.utils import data
import os
import scipy
from scipy.io import loadmat
from skimage.color import rgb2lab
from skimage.util import img_as_float
from skimage import io
from skimage import color
import numpy as np
from random import Random
from scipy import interpolate
from util import convert_index
import matplotlib.pyplot as plt
import torch
import skimage
from PIL import Image

RAND_SEED = 2021
myrandom = Random()


def convert_label(label, num=50):
    problabel = np.zeros((1, num, label.shape[0], label.shape[1])).astype(np.float32)

    ct = 0
    for t in np.unique(label).tolist():
        if ct >= num:
            # print(np.unique(label).shape)
            break
            # raise IOError
        else:
            problabel[:, ct, :, :] = (label == t)
        ct = ct + 1

    label2 = np.squeeze(np.argmax(problabel, axis=1))

    return label2, problabel


def transform_and_get_image(im, max_spixels, out_size):
    height = im.shape[0]
    width = im.shape[1]

    out_height = out_size[0]
    out_width = out_size[1]

    pad_height = out_height - height
    pad_width = out_width - width
    im = np.lib.pad(im, ((0, pad_height), (0, pad_width), (0, 0)), 'constant',
                    constant_values=-10)
    im = np.expand_dims(im, axis=0)
    return im


def get_spixel_init(num_spixels, img_width, img_height):
    """
    :return each pixel belongs to which pixel
    """

    k = num_spixels
    k_w = int(np.floor(np.sqrt(k * img_width / img_height)))
    k_h = int(np.floor(np.sqrt(k * img_height / img_width)))

    spixel_height = img_height / (1. * k_h)
    spixel_width = img_width / (1. * k_w)

    h_coords = np.arange(-spixel_height / 2. - 1, img_height + spixel_height - 1,
                         spixel_height)
    w_coords = np.arange(-spixel_width / 2. - 1, img_width + spixel_width - 1,
                         spixel_width)
    spix_values = np.int32(np.arange(0, k_w * k_h).reshape((k_h, k_w)))
    spix_values = np.pad(spix_values, 1, 'symmetric')
    f = interpolate.RegularGridInterpolator((h_coords, w_coords), spix_values, method='nearest')

    all_h_coords = np.arange(0, img_height, 1)
    all_w_coords = np.arange(0, img_width, 1)
    all_grid = np.array(np.meshgrid(all_h_coords, all_w_coords, indexing='ij'))
    all_points = np.reshape(all_grid, (2, img_width * img_height)).transpose()

    spixel_initmap = f(all_points).reshape((img_height, img_width))

    feat_spixel_initmap = spixel_initmap
    return [spixel_initmap, feat_spixel_initmap, k_w, k_h]


def transform_and_get_spixel_init(max_spixels, out_size):
    out_height = out_size[0]
    out_width = out_size[1]

    spixel_init, feat_spixel_initmap, k_w, k_h = \
        get_spixel_init(max_spixels, out_width, out_height)
    spixel_init = spixel_init[None, None, :, :]
    feat_spixel_initmap = feat_spixel_initmap[None, None, :, :]

    return spixel_init, feat_spixel_initmap, k_h, k_w


def get_rand_scale_factor():
    rand_factor = np.random.normal(1, 0.75)

    s_factor = np.min((3.0, rand_factor))
    s_factor = np.max((0.75, s_factor))

    return s_factor


def scale_image(im, s_factor):
    s_img = scipy.ndimage.zoom(im, (s_factor, s_factor, 1), order=1)

    return s_img


def scale_label(label, s_factor):
    s_label = scipy.ndimage.zoom(label, (s_factor, s_factor), order=0)

    return s_label


def PixelFeature(img, color_scale=None, pos_scale=None, type=None):
    b, h, w, c = img.shape
    feat = img * color_scale
    if type == 'RGB_AND_POSITION':  # yxrcb
        x_axis = np.arange(0, w, 1)
        y_axis = np.arange(0, h, 1)
        x_mesh, y_mesh = np.meshgrid(x_axis, y_axis)
        yx = np.stack([y_mesh, x_mesh], axis=-1)
        yx_scaled = yx * pos_scale
        yx_scaled = np.repeat(yx_scaled[np.newaxis], b, axis=0)
        feat = np.concatenate([yx_scaled, feat], axis=-1)
    return feat


class Dataset_Train(data.Dataset):
    def __init__(self, num_spixel, file_names=None, root=None, patch_size=None):
        self.patch_size = patch_size
        # self.width = width
        self.num_spixel = num_spixel
        self.out_types = ['left_img', 'right_img', 'spixel_init', 'feat_spixel_init', 'label', 'problabel']

        self.file_names = file_names
        self.root = root
        self.left_dir = root + 'left/'
        self.right_dir = root + 'right/'
        self.seg_dir = root + 'seg/'

        # init pixel-spixel index
        self.out_spixel_init, self.feat_spixel_init, self.spixels_h, self.spixels_w = \
            transform_and_get_spixel_init(self.num_spixel, [patch_size[0], patch_size[1]])
        self.init, self.cir, self.p2sp_index_, self.invisible = convert_index(self.spixels_w,
                                                                              self.spixels_w * self.spixels_h,
                                                                              self.feat_spixel_init)
        self.invisible = self.invisible.astype(np.float)

    def __getitem__(self, item):
        # print('start getting images')
        # 获取左右图
        left_img_path = self.left_dir + self.file_names[item]
        # right_img_path = self.right_dir + self.file_names[item]
        right_img_path = self.right_dir + self.file_names[item]
        # seg_img_path = self.seg_dir + self.file_names[item]
        seg_img_path = self.seg_dir + self.file_names[item]
        im_left = Image.open(left_img_path)
        im_right = Image.open(right_img_path)
        im_left = np.asarray(im_left)
        im_right = np.asarray(im_right)

        h, w, _ = im_left.shape

        # 获取gt
        gtseg = io.imread(seg_img_path)

        # print('start augmentation images')
        # if np.random.uniform(0, 1) > 0.5:
        #     im_left = left_image[:, ::-1, ...]
        #     im_right = right_image[:, ::-1, ...]
        #     gtseg = gtseg[:, ::-1]

        if self.patch_size is None:
            raise Exception('not define the output size')
        else:
            out_height = self.patch_size[0]
            out_width = self.patch_size[1]
        if out_height > h:
            raise Exception("Patch size is greater than image size")

        if out_width > w:
            raise Exception("Patch size is greater than image size")

        start_row = myrandom.randint(0, h - out_height)
        start_col = myrandom.randint(0, w - out_width)
        # print(start_row, start_col)
        im_cropped_left = im_left[start_row: start_row + out_height, start_col: start_col + out_width, :]
        im_cropped_right = im_right[start_row: start_row + out_height, start_col: start_col + out_width, :]
        # print('start asfloat')
        im_cropped_left = img_as_float(im_cropped_left)
        im_cropped_right = img_as_float(im_cropped_right)
        # print('start rgb2lab')
        im_cropped_left = rgb2lab(im_cropped_left)
        im_cropped_right = rgb2lab(im_cropped_right)
        out_img_left = transform_and_get_image(im_cropped_left, self.num_spixel, [out_height, out_width])
        out_img_right = transform_and_get_image(im_cropped_right, self.num_spixel, [out_height, out_width])
        # add xy information
        out_img_left = PixelFeature(out_img_left, color_scale=0.26, pos_scale=0.125, type='RGB_AND_POSITION')
        out_img_right = PixelFeature(out_img_right, color_scale=0.26, pos_scale=0.125, type='RGB_AND_POSITION')

        gtseg_cropped = gtseg[start_row: start_row + out_height, start_col: start_col + out_width]
        # print('start converting labels')
        label_cropped, problabel_cropped = convert_label(gtseg_cropped, num=34)

        # print('start building inputs')
        inputs = {}
        for in_name in self.out_types:
            # 图片形式
            if in_name == 'left_img':
                inputs['left_img'] = np.transpose(out_img_left[0], [2, 0, 1]).astype(np.float32)
            if in_name == 'right_img':
                inputs['right_img'] = np.transpose(out_img_right[0], [2, 0, 1]).astype(np.float32)
            # GT中的Label
            if in_name == 'label':
                label_cropped = np.expand_dims(np.expand_dims(label_cropped, axis=0), axis=0)
                inputs['label'] = label_cropped[0]
            # 将GT中的Label转换为one-hot形式
            if in_name == 'problabel':
                inputs['problabel'] = problabel_cropped[0]
            # 初始化的超像素
            if in_name == 'spixel_init':
                inputs['spixel_init'] = self.out_spixel_init[0].astype(np.float32)
            # 初始化的超像素
            if in_name == 'feat_spixel_init':
                inputs['feat_spixel_init'] = self.feat_spixel_init[0].astype(np.float32)

        '''
        inputs:见上述注释
        spixels_h,spixels_w:超像素在高和宽上的数量
        init: x坐标，y坐标，第三个难以用语言形容...
        p2sp_index: 每个像素对应的周围9个超像素
        invisible:处理边缘处的超像素（周边的超像素是否为9个）（0为存在，1为不存在）
        '''
        # print('data loading over')
        return inputs, self.spixels_h, self.spixels_w, self.init, self.cir, self.p2sp_index_, self.invisible, \
               self.file_names[item]

    def __len__(self):
        return len(self.file_names)


class Dataset_Test(data.Dataset):
    def __init__(self, num_spixel, file_names=None, root=None):
        self.num_spixel = num_spixel
        self.out_types = ['left_img', 'right_img', 'spixel_init', 'feat_spixel_init', 'label', 'problabel']

        self.file_names = file_names
        self.root = root
        self.left_dir = root + 'left/'
        self.right_dir = root + 'right/'
        self.seg_dir = root + 'seg/'

    def __getitem__(self, item):
        left_img_path = self.left_dir + self.file_names[item]
        # right_img_path = self.right_dir + self.file_names[item]
        right_img_path = self.right_dir + self.file_names[item].replace("left", "right")
        # seg_img_path = self.seg_dir + self.file_names[item]
        seg_img_path = self.seg_dir + self.file_names[item].replace('leftImg8bit', 'gtFine_labelIds')
        left_image = io.imread(left_img_path) / 255.0
        right_image = io.imread(right_img_path) / 255.0
        im_left = rgb2lab(left_image)
        im_right = rgb2lab(right_image)
        h, w, _ = im_left.shape

        gtseg = io.imread(seg_img_path)

        k = self.num_spixel
        k_w = int(np.floor(np.sqrt(k * w / h)))
        k_h = int(np.floor(np.sqrt(k * h / w)))
        spixel_height = h / (1. * k_h)
        spixel_width = w / (1. * k_w)

        out_height = int(np.ceil(spixel_height) * k_h)
        out_width = int(np.ceil(spixel_width) * k_w)

        out_img_left = transform_and_get_image(im_left, self.num_spixel, [out_height, out_width])
        # add xy information
        pos_scale = 2.5 * max(k_h / out_height, k_w / out_width)
        out_img_left = PixelFeature(out_img_left, color_scale=0.26, pos_scale=pos_scale, type='RGB_AND_POSITION')
        out_img_right = transform_and_get_image(im_right, self.num_spixel, [out_height, out_width])
        out_img_right = PixelFeature(out_img_right, color_scale=0.26, pos_scale=pos_scale, type='RGB_AND_POSITION')

        gtseg_ = np.ones_like(out_img_left[0, :, :, 0]) * 49
        gtseg_[:h, :w] = gtseg
        label_cropped, problabel_cropped = convert_label(gtseg_)

        self.out_spixel_init, self.feat_spixel_init, self.spixels_h, self.spixels_w = \
            transform_and_get_spixel_init(self.num_spixel, [out_height, out_width])
        self.init, self.cir, self.p2sp_index_, self.invisible = convert_index(self.spixels_w,
                                                                              self.spixels_w * self.spixels_h,
                                                                              self.feat_spixel_init)
        self.invisible = self.invisible.astype(np.float)

        inputs = {}
        for in_name in self.out_types:
            if in_name == 'left_img':
                inputs['left_img'] = np.transpose(out_img_left[0], [2, 0, 1]).astype(np.float32)
            if in_name == 'right_img':
                inputs['right_img'] = np.transpose(out_img_right[0], [2, 0, 1]).astype(np.float32)
            if in_name == 'spixel_init':
                inputs['spixel_init'] = self.out_spixel_init[0].astype(np.float32)
            if in_name == 'feat_spixel_init':
                inputs['feat_spixel_init'] = self.feat_spixel_init[0].astype(np.float32)
            if in_name == 'label':
                label_cropped = np.expand_dims(np.expand_dims(label_cropped, axis=0), axis=0)
                inputs['label'] = label_cropped[0]
            if in_name == 'problabel':
                inputs['problabel'] = problabel_cropped[0]

        return inputs, self.spixels_h, self.spixels_w, self.init, self.cir, self.p2sp_index_, self.invisible, \
               self.file_names[item], h, w

    def __len__(self):
        return len(self.file_names)


# SSN Dataset
# class Dataset(data.Dataset):
#     def __init__(self, num_spixel, root=None, patch_size=None, dtype='train'):
#         self.patch_size = patch_size
#         # self.width = width
#         self.num_spixel = num_spixel
#         self.out_types = ['img', 'spixel_init', 'feat_spixel_init', 'label', 'problabel']
#
#         self.root = root
#         self.dtype = dtype
#         self.data_dir = os.path.join(self.root, 'BSR', 'BSDS500', 'data')
#
#         self.split_list = open(os.path.join(root, dtype + '.txt')).readlines()
#         self.img_dir = os.path.join(self.data_dir, 'images', self.dtype)
#         self.gt_dir = os.path.join(self.data_dir, 'groundTruth', self.dtype)
#
#         # init pixel-spixel index
#         self.out_spixel_init, self.feat_spixel_init, self.spixels_h, self.spixels_w = \
#             transform_and_get_spixel_init(self.num_spixel, [patch_size[0], patch_size[1]])
#         self.init, self.cir, self.p2sp_index_, self.invisible = convert_index(self.spixels_w,
#                                                                               self.spixels_w * self.spixels_h,
#                                                                               self.feat_spixel_init)
#         self.invisible = self.invisible.astype(np.float)
#
#     def __getitem__(self, item):
#         img_name = self.split_list[item].rstrip('\n')
#         e = io.imread(os.path.join(self.img_dir, img_name + '.jpg'))
#         image = img_as_float(io.imread(os.path.join(self.img_dir, img_name + '.jpg')))
#         s_factor = get_rand_scale_factor()
#         image = scale_image(image, s_factor)
#         im = rgb2lab(image)
#         h, w, _ = im.shape
#
#         gtseg_all = loadmat(os.path.join(self.gt_dir, img_name + '.mat'))
#         t = np.random.randint(0, len(gtseg_all['groundTruth'][0]))
#         gtseg = gtseg_all['groundTruth'][0][t][0][0][0]
#         gtseg = scale_label(gtseg, s_factor)
#
#         if np.random.uniform(0, 1) > 0.5:
#             im = im[:, ::-1, ...]
#             gtseg = gtseg[:, ::-1]
#
#         if self.patch_size == None:
#             raise ('not define the output size')
#         else:
#             out_height = self.patch_size[0]
#             out_width = self.patch_size[1]
#
#         if out_height > h:
#             raise ("Patch size is greater than image size")
#
#         if out_width > w:
#             raise ("Patch size is greater than image size")
#
#         start_row = myrandom.randint(0, h - out_height)
#         start_col = myrandom.randint(0, w - out_width)
#         im_cropped = im[start_row: start_row + out_height,
#                      start_col: start_col + out_width, :]
#         out_img = transform_and_get_image(im_cropped, self.num_spixel, [out_height, out_width])
#         # add xy information
#         out_img = PixelFeature(out_img, color_scale=0.26, pos_scale=0.125, type='RGB_AND_POSITION')
#
#         gtseg_cropped = gtseg[start_row: start_row + out_height,
#                         start_col: start_col + out_width]
#         label_cropped, problabel_cropped = convert_label(gtseg_cropped)
#
#         inputs = {}
#         for in_name in self.out_types:
#             if in_name == 'img':
#                 inputs['img'] = np.transpose(out_img[0], [2, 0, 1]).astype(np.float32)
#             if in_name == 'spixel_init':
#                 inputs['spixel_init'] = self.out_spixel_init[0].astype(np.float32)
#             if in_name == 'feat_spixel_init':
#                 inputs['feat_spixel_init'] = self.feat_spixel_init[0].astype(np.float32)
#             if in_name == 'label':
#                 label_cropped = np.expand_dims(np.expand_dims(label_cropped, axis=0), axis=0)
#                 inputs['label'] = label_cropped[0]
#             if in_name == 'problabel':
#                 inputs['problabel'] = problabel_cropped[0]
#
#         return inputs, self.spixels_h, self.spixels_w, self.init, self.cir, self.p2sp_index_, self.invisible
#
#     def __len__(self):
#         return len(self.split_list)

if __name__ == '__main__':
    dataloader = data.DataLoader(
        Dataset_Train(num_spixel=100, patch_size=[200, 200], root='../../data/KITTI/', file_names=['000000_10.png']),
        batch_size=1, shuffle=True)
