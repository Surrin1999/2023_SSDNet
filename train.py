from tqdm import tqdm
from config import CFG
from dataset.dataset import *
from dataset.train_valid_split import *
from model.model import *
from loss import *
import argparse
from skimage.segmentation._slic import _enforce_label_connectivity_cython
from metric import compute_asa
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
from IPython import display


class loss_logger():
    def __init__(self):
        self.loss = 0
        self.loss1 = 0
        self.loss2 = 0
        self.loss3 = 0
        self.count = 0

    def add(self, l, l1, l2, l3):
        self.loss *= self.count
        self.loss1 *= self.count
        self.loss2 *= self.count
        self.loss3 *= self.count
        self.loss += l
        self.loss1 += l1
        self.loss2 += l2
        self.loss3 += l3
        self.count += 1

    def ave(self):
        self.loss /= self.count
        self.loss1 /= self.count
        self.loss2 /= self.count
        self.loss3 /= self.count

    def clear(self):
        self.__init__()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--sp_num_train', type=int, default=CFG.sp_num_train)
    parser.add_argument('--sp_num_valid', type=int, default=CFG.sp_num_valid)
    parser.add_argument('--root', type=str, default=CFG.root)
    parser.add_argument('--train_bs', type=int, default=CFG.train_bs)
    parser.add_argument('--valid_bs', type=int, default=CFG.valid_bs)
    parser.add_argument('--num_iter_train', type=int, default=CFG.num_iter_train)
    parser.add_argument('--num_iter_valid', type=int, default=CFG.num_iter_valid)
    parser.add_argument('--epoch', type=int, default=CFG.epoch)
    parser.add_argument('--save_ckpt_path', type=str, default=CFG.save_ckpt_path)
    parser.add_argument('--lr', type=float, default=CFG.lr)

    args = parser.parse_args()

    return args


def train_one_epoch(epoch, model, optimizer, dataloader, device, total_iter):
    logger = loss_logger()
    criten = Loss()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for iter, [inputs, num_h, num_w, init_index, cir_index,
               p2sp_index_, invisible, filename] in pbar:
        with torch.autograd.set_detect_anomaly(True):
            total_iter = total_iter + 1
            left_img = inputs['left_img'].to(device)
            right_img = inputs['right_img'].to(device)
            label = inputs['label'].to(device)
            problabel = inputs['problabel'].to(device)
            if iter == 0 or iter == 18:
                num_h_i = num_h.to(device)
                num_w_i = num_w.to(device)
                init_index_i = [x.to(device) for x in init_index]
                cir_index_i = [x.to(device) for x in cir_index]
                p2sp_index__i = p2sp_index_.to(device)
                invisible_i = invisible.to(device)
            b, c, h, w = left_img.shape
            l = left_img[:, 2:, :, :].permute(0, 2, 3, 1).cpu().numpy()
            r = right_img[:, 2:, :, :].permute(0, 2, 3, 1).cpu().numpy()
            for i in range(b):
                l[i] = lab2rgb(l[i] / 0.26)
                r[i] = lab2rgb(r[i] / 0.26)
            l = torch.tensor(l).cuda().permute(0, 3, 1, 2)
            r = torch.tensor(r).cuda().permute(0, 3, 1, 2)

            # print('start processing...')
            recon_feat2, recon_label, new_spix_indices, \
            (M_right_to_left, M_left_to_right), \
            (M_left_right_left, M_right_left_right), \
            (V_left_to_right, V_right_to_left) = model(l, r, left_img, right_img, p2sp_index_,
                                                       invisible_i, init_index_i, cir_index_i,
                                                       problabel, num_h_i, num_w_i, device)
            # recon_feat2, recon_label, new_spix_indices = model(l, r, left_img, right_img, p2sp_index_,
            #                                            invisible, init_index, cir_index,
            #                                            problabel, num_h, num_w, device)
            # print('processing over...')
            loss, loss_p_c, loss_label = criten(recon_feat2, left_img, recon_label, label)
            L_photometric = photometric_loss(M_right_to_left, M_left_to_right,
                                             V_left_to_right, V_right_to_left,
                                             l, r, b, h, w, c - 2)
            # if iter % 1 == 0:
            #     display.clear_output(wait=True)
            #     # v = V_left_to_right[0][0].cpu().numpy()
            #     l = left_img[0][2:].permute(1, 2, 0).cpu().numpy()
            #     l = lab2rgb(l / 0.26)
            #     r = right_img[0][2:].permute(1, 2, 0).cpu().numpy()
            #     r = lab2rgb(r / 0.26)
            #     # generate the superpixel map
            #     new_spix_indices = new_spix_indices[:, :h, :w].contiguous()
            #     spix_index = new_spix_indices.cpu().numpy()[0]
            #     spix_index = spix_index.astype(int)
            #
            #     # segment_size = (h * w) / (int(num_h * num_w) * 1.0)
            #     min_size = int(0.06 * 400)
            #     max_size = int(3 * 400)
            #     spix_index = \
            #         _enforce_label_connectivity_cython(spix_index[np.newaxis, :, :].astype(np.int64),
            #                                            min_size, max_size)[0]
            #     spixel_image = get_spixel_image(l, spix_index)
            #
            #     # m = M_right_to_left[0][100].cpu().detach().numpy()
            #     plt.subplot(1, 2, 1)
            #     plt.imshow(l)
            #     plt.subplot(1, 2, 2)
            #     plt.imshow(spixel_image)
            #     # plt.subplot(2, 2, 3)
            #     # plt.imshow(v)
            #     # plt.subplot(2, 2, 4)
            #     # plt.imshow(m)
            #     plt.show()

            loss_stereo = L_photometric
            loss += loss_stereo

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if total_iter % 2000 == 0 and total_iter < 8000:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] / 2
            if total_iter == 8000:
                for g in optimizer.param_groups:
                    g['lr'] = 0.00002

            # compute the logger
            logger.add(loss, loss_p_c, loss_label, loss_stereo)
            logger.ave()

            for g in optimizer.param_groups:
                lr = g['lr']
                break
            # set the description of progressing bar
            description = 'epoch:%d, lr:%5f, floss:%.3f, loss_pc:%.8f, loss_label:%.3f, loss_stereo:%.3f' % (
                epoch, lr, logger.loss, logger.loss1, logger.loss2, logger.loss3)
            pbar.set_description(description)
            # clear redundant variables and release video memory
            torch.cuda.empty_cache()

    return logger.loss, model, optimizer, total_iter


def valid_one_epoch(epoch, model, dataloader, device):
    logger = loss_logger()
    criten = Loss()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    asa_sum = 0
    for iter, [inputs, num_h, num_w, init_index, cir_index,
               p2sp_index_, invisible, file_name] in pbar:
        with torch.no_grad():
            left_img = inputs['left_img'].to(device)
            right_img = inputs['right_img'].to(device)
            label = inputs['label'].to(device)
            problabel = inputs['problabel'].to(device)
            num_h = num_h.to(device)
            num_w = num_w.to(device)
            init_index = [x.to(device) for x in init_index]
            cir_index = [x.to(device) for x in cir_index]
            p2sp_index_ = p2sp_index_.to(device)
            invisible = invisible.to(device)
            b, c, h, w = left_img.shape

            recon_feat2, recon_label, (M_right_to_left, M_left_to_right), \
            (M_left_right_left, M_right_left_right), \
            (V_left_to_right, V_right_to_left) = model(left_img, right_img, p2sp_index_,
                                                       invisible, init_index, cir_index,
                                                       problabel, num_h, num_w, device)

            # compute the loss
            loss, loss_p_c, loss_label = criten(recon_feat2, left_img, recon_label, label)
            L_photometric = photometric_loss(M_right_to_left, M_left_to_right,
                                             V_left_to_right, V_right_to_left,
                                             left_img, right_img, b, h, w, c)
            loss_stereo = L_photometric
            loss += 0.005 * loss_stereo
            logger.add(loss, loss_p_c, loss_label, loss_stereo)
            logger.ave()

            # generate the superpixel map
            new_spix_indices = new_spix_indices[:, :h, :w].contiguous()
            spix_index = new_spix_indices.cpu().numpy()[0]
            spix_index = spix_index.astype(int)

            segment_size = (h * w) / (int(num_h * num_w) * 1.0)
            min_size = int(0.06 * segment_size)
            max_size = int(3 * segment_size)
            spix_index = \
                _enforce_label_connectivity_cython(spix_index[np.newaxis, :, :].astype(np.int64),
                                                   min_size, max_size)[0]
            # calculate the asa score of valid images
            asa = compute_asa(spix_index, label[0][0])
            asa_sum = asa_sum + asa
            asa_avg = asa_sum / (iter + 1)

            # set the description of progressing bar
            description = 'epoch:%d, loss:%.3f, asa:%.3f' % (epoch, logger.loss, asa_avg)
            pbar.set_description(description)

    return asa_avg * 100.0


def main_loop(args):
    # 150 images for train and 50 images for valid
    train_file_names, valid_file_names = get_train_valid()
    dataloader_train = data.DataLoader(Dataset_Train(args.sp_num_train,
                                                     file_names=train_file_names,
                                                     root=args.root, patch_size=[200, 200]),
                                       batch_size=args.train_bs, shuffle=True)
    # dataloader_valid = data.DataLoader(Dataset_Test(args.sp_num_valid,
    #                                                 file_names=valid_file_names,
    #                                                 root=args.root),
    #                                    batch_size=args.valid_bs, shuffle=True)
    model = StereoSpixelNet(num_iter=args.num_iter_train, is_trainning=True)

    min_loss = 100

    device = torch.device('cpu')
    if torch.cuda.is_available():
        model.cuda()
        device = torch.device('cuda')

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_iter = 0
    for epoch in range(args.epoch):
        model.train()
        loss, model, optim, total_iter = train_one_epoch(epoch, model, optim, dataloader_train, device, total_iter)
        torch.save(model.state_dict(), args.save_ckpt_path +
                   f'KITTI_E%d_L%.3f.pth' % (epoch, loss))

        if loss < min_loss:
            min_loss = loss
            print('Min Loss: {}'.format(min_loss))


if __name__ == '__main__':
    main_loop(get_args())
