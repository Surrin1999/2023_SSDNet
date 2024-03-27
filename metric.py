import torch


def compute_asa(spixel_map, gt):
    asa_num = 0
    sp_num = int(torch.max(spixel_map) + 1)
    gt_num = int(torch.max(gt) + 1)

    sumim = 1 + gt + spixel_map * gt_num
    hs = torch.histc(sumim.float(), bins=int(sp_num * gt_num), min=1, max=sp_num * gt_num)
    hs = hs.view(sp_num, gt_num)
    for i in range(sp_num):
        asa_num = asa_num + torch.max(hs[i]).numpy()
    asa = asa_num / (gt.shape[0] * gt.shape[1])

    return asa


def compute_undersegmentation_error(spixel_map, gt):
    use_num = 0
    sp_num = int(torch.max(spixel_map) + 1)
    gt_num = int(torch.max(gt) + 1)

    sumim = 1 + gt + spixel_map * gt_num
    hs = torch.histc(sumim.float(), bins=int(sp_num * gt_num), min=1, max=sp_num * gt_num)
    hs = hs.view(sp_num, gt_num)
    for j in range(gt_num):
        for i in range(sp_num):
            resj = torch.sum(hs[i])
            gtjresj = hs[i][j]
            use_num = use_num + torch.min(gtjresj, resj - gtjresj)
    use = use_num / (gt.shape[0] * gt.shape[1])
    return use.numpy()

# def compute_bdry(spixel_map, gt, bdry, nthresh, maxDist=0.0075, thinpb=1):
