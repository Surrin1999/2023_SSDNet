import torch
from tqdm import tqdm
from config import CFG
from dataset.dataset import *
from dataset.train_valid_split import *
from model.model import *
from metric import *
from skimage.segmentation._slic import _enforce_label_connectivity_cython
import argparse
import skimage.io as io
import scipy.io as scio
import os


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--sp_num_test', type=int, default=CFG.sp_num_test)
    parser.add_argument('--root', type=str, default=CFG.root)
    parser.add_argument('--test_bs', type=int, default=CFG.test_bs)
    parser.add_argument('--num_iter_test', type=int, default=CFG.num_iter_test)
    parser.add_argument('--save_mat_path', type=str, default=CFG.save_mat_path)
    args = parser.parse_args()

    return args


def test_image(model, dataloader, device, sp_num, save_mat_path):
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    asa_sum = 0
    use_sum = 0

    save_path = 'mat/{}/'.format(sp_num)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    for iter, [inputs, num_h, num_w, init_index, cir_index,
               p2sp_index_, invisible, file_name, h, w] in pbar:
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
            b, c, lh, lw = left_img.shape
            l = left_img[:, 2:, :, :].permute(0, 2, 3, 1).cpu().numpy()
            r = right_img[:, 2:, :, :].permute(0, 2, 3, 1).cpu().numpy()
            for i in range(b):
                l[i] = lab2rgb(l[i] / 0.26)
                r[i] = lab2rgb(r[i] / 0.26)
            l = torch.tensor(l).cuda().permute(0, 3, 1, 2)
            r = torch.tensor(r).cuda().permute(0, 3, 1, 2)

            recon_feat2, new_spix_indices = model(l, r, left_img, right_img, p2sp_index_,
                                                  invisible, init_index, cir_index,
                                                  problabel, num_h, num_w, device)

            # generate the superpixel map
            new_spix_indices = new_spix_indices.contiguous()
            spix_index = new_spix_indices.cpu().numpy()[0]
            spix_index = spix_index.astype(int)

            segment_size = (lh * lw) / (int(num_h * num_w) * 1.0)
            min_size = int(0.06 * segment_size)
            max_size = int(3 * segment_size)
            spix_index = \
                _enforce_label_connectivity_cython(spix_index[np.newaxis, :, :].astype(np.int64), min_size, max_size)[
                    0][:h, :w]

            # calculate the asa score of valid images
            asa = compute_asa(torch.tensor(spix_index), label[0][0][:h, :w].cpu().detach())
            asa_sum = asa_sum + asa
            asa_avg = asa_sum / (iter + 1)

            use = compute_undersegmentation_error(torch.tensor(spix_index), label[0][0][:h, :w].cpu().detach())
            use_sum = use_sum + use
            use_avg = use_sum / (iter + 1)

            l = left_img[0][2:].permute(1, 2, 0).cpu().numpy()
            l = lab2rgb(l / 0.26)

            # spixel_image = get_spixel_image(l, spix_index)
            # save_path = 'result/KITTI/' + str(CFG.sp_num_test) + '/' + str(iter) + '.png'
            # io.imsave(save_path, spixel_image)

            # save the result into .mat format
            example = scio.loadmat('example.mat')
            example['segs'][0][0] = spix_index[:h, :w]

            file_dir = save_mat_path + str(sp_num) + '/'
            if not os.path.exists(file_dir):
              os.makedirs(file_dir)
            file_name = file_name[0].split('\\')[-1][:-4] + ".mat"
            scio.savemat(file_dir + file_name, {'segs':example['segs']})

            # set the description of progressing bar
            description = 'Sp num: %d - ASA:%.3f, USE:%.3f' % (sp_num, asa_avg * 100.0, use_avg * 100)
            pbar.set_description(description)


def test_main_loop(args):
    test_file_names = get_test()

    for sp_num in args.sp_num_test:
        # load the model
        model = StereoSpixelNet(num_iter=args.num_iter_test, is_trainning=False)
        device = torch.device('cpu')
        if torch.cuda.is_available():
            model.cuda()
            device = torch.device('cuda')
        # load the weight of model B0
        model.load_state_dict(torch.load("checkpoint/KITTI/B0/model.pth"))
        model.eval()

        # load the test data
        dataloader_test = data.DataLoader(Dataset_Test(sp_num, file_names=test_file_names, root=args.root),
                                          batch_size=args.test_bs)

        test_image(model, dataloader_test, device, sp_num, args.save_mat_path)


if __name__ == '__main__':
    test_main_loop(get_args())
