class CFG:
    train_bs = 8
    valid_bs = 8
    test_bs = 1
    epoch = 1000
    lr = 0.0002
    train_size = 200
    # root = 'data/Cityscapes/'
    root = 'data/KITTI/'
    sp_num_train = 100
    sp_num_valid = 200
    sp_num_test = [200, 300, 400, 500, 620, 680, 780]
    num_iter_train = 5
    num_iter_valid = 10
    num_iter_test = 10
    save_ckpt_path = 'checkpoint/KITTI/B0/'
    save_mat_path = 'result/KITTI/'
