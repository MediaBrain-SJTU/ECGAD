import os
import random
import argparse
import time
import math
import torch
import numpy as np
from tqdm import tqdm
from utils import generate_trend, normalize
from model import MCF
from dataloader import TestSet, PixelTestSet
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter
import copy
import warnings

warnings.filterwarnings("ignore")
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def main():
    parser = argparse.ArgumentParser(description='ECG anomaly detection, test')
    parser.add_argument('--data_path', type=str, default='data/')
    parser.add_argument('--dims', type=int, default=12, help='dimension of the input data')
    parser.add_argument('--load_model', type=int, default=1, help='0 for retrain, 1 for load model')
    parser.add_argument('--load_path', type=str, default='ckpt/mymodel.pt')
    parser.add_argument('--mask_ratio', type=int, default=30, help='mask ratio for self-restoration')
    parser.add_argument('--seed', type=int, default=668, help='manual seed')
    parser.add_argument('--hidden', type=int, default=50, help='hidden dimension')

    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.randint(1, 10000)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)
    print("args: ", args)


    # load dataset
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    dtset = TestSet(folder=args.data_path)
    test_loader = torch.utils.data.DataLoader(dtset, batch_size=1, shuffle=False, **kwargs)
    labels = np.load(os.path.join(args.data_path, 'label.npy'))

    dtset = PixelTestSet(folder=args.data_path)
    pixeltest_loader = torch.utils.data.DataLoader(dtset, batch_size=1, shuffle=False, **kwargs)
    pixel_labels = np.load(os.path.join(args.data_path, 'benchmark_label.npy'))


    # load model
    model = MCF(enc_in=args.dims, hidden=args.hidden).to(device)
    if args.load_model == 1:
        checkpoint = torch.load(args.load_path)
        model.load_state_dict(checkpoint['model_state_dict'])


    # test
    detection_test(args, model, test_loader, labels)
    localization_test(args, model, pixeltest_loader, pixel_labels)


def detection_test(args, model, test_loader, labels):
    torch.zero_grad = True
    model.eval()

    patch_interval = 4800 // args.mask_ratio
    cut_length = 480 * args.mask_ratio // 100
    cutidx_list = [20, 70, 120, 170, 220]

    result = []
    for i, (r_index, ori_global) in tqdm(enumerate(test_loader)):

        ori_global = ori_global.float().cuda()
        global_ecg = ori_global[:,100:4900:]
        trend = generate_trend(global_ecg)

        global_length = global_ecg.shape[1]
        _, idx_length = r_index.shape

        local_division_result = []
        for r_idx in range(idx_length):
            r_index_value = r_index[0][r_idx]
            if r_index_value>200 and r_index_value<4800-400:
                local_ecg = ori_global[:,r_index_value-140:r_index_value+340,:]
                local_ecg = normalize(local_ecg)

                instance_result = []
                for j in range(100//args.mask_ratio):

                    # mask on global ecg
                    mask_global = copy.deepcopy(global_ecg)
                    mask = torch.zeros((1,global_length,1), dtype=torch.bool).cuda()
                    for k in range(args.mask_ratio):
                        cut_idx = 48*j + patch_interval*k
                        mask[:,cut_idx:cut_idx+48] = 1
                    mask_global = torch.mul(mask_global, ~mask)

                    # mask on local ecg
                    mask_local = copy.deepcopy(local_ecg)
                    cut_idx = cutidx_list[j%5]
                    cut_length = 100
                    mask_local[:, cut_idx:cut_idx+cut_length ,:] = 0

                    (gen_global, global_var), (gen_local, local_var), gen_trend = model(mask_global, mask_local, trend)

                    global_err = (gen_global - global_ecg) ** 2
                    local_err = (gen_local - local_ecg) ** 2
                    trend_err = (gen_trend - global_ecg) ** 2

                    l_global = torch.mean(torch.exp(-global_var)*global_err)
                    l_local = torch.mean(torch.exp(-local_var)*local_err)
                    l_trend = torch.mean(trend_err)

                    final_loss = l_global + l_local + l_trend

                    final_loss = final_loss.detach().cpu().numpy()
                    instance_result.append(final_loss)

                tmp_instance_result = np.asarray(instance_result)
                local_division_result.append(tmp_instance_result.mean())
            else:
                continue

        local_division_result = np.array(local_division_result)
        result.append(local_division_result.mean())

    scores = np.asarray(result)
    test_labels = np.array(labels).astype(int)

    max_anomaly_score = scores.max()
    min_anomaly_score = scores.min()
    scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

    auc_result = roc_auc_score(test_labels, scores)

    print(("Detection AUC: ", round(auc_result, 3)))
    return auc_result


def localization_test(args, model, test_loader, labels):
    torch.zero_grad = True
    model.eval()

    patch_interval = 4800 // args.mask_ratio
    cut_length = 480 * args.mask_ratio // 100
    cutidx_list = [20, 70, 120, 170, 220]

    result = []

    for i, (r_index, ori_global) in tqdm(enumerate(test_loader)):

        ori_global = ori_global.float().cuda()
        global_ecg = ori_global[:,100:4900:]
        trend = generate_trend(global_ecg)

        global_length = global_ecg.shape[1]
        _, idx_length = r_index.shape

        loss_to_draw = None
        beat_division_result = []
        valid_rpeak_cnt = 0
        local_loss_all = torch.zeros((1,5000,12)).cuda()
        for r_idx in range(idx_length):
            r_index_value = r_index[0][r_idx]
            if r_index_value>200 and r_index_value<4800-400:
                valid_rpeak_cnt += 1
                local_ecg = ori_global[:,r_index_value-140:r_index_value+340,:]
                local_ecg = normalize(local_ecg)
                
                sub_loss_to_draw = None
                for j in range(100//args.mask_ratio):
                    # mask on global ecg
                    mask_global = copy.deepcopy(global_ecg)
                    mask = torch.zeros((1,global_length,1), dtype=torch.bool).cuda()
                    for k in range(args.mask_ratio):
                        cut_idx = 48*j + patch_interval*k
                        mask[:,cut_idx:cut_idx+48] = 1
                    mask_global = torch.mul(mask_global, ~mask)

                    # mask on local ecg
                    mask_local = copy.deepcopy(local_ecg)
                    cut_idx = cutidx_list[j%5]
                    cut_length = 100
                    mask_local[:, cut_idx:cut_idx+cut_length ,:] = 0

                    (gen_global, global_var), (gen_local, local_var), gen_trend = model(mask_global, mask_local, trend)

                    global_err = (gen_global - global_ecg) ** 2
                    local_err = (gen_local - local_ecg) ** 2
                    trend_err = (gen_trend - global_ecg) ** 2

                    if sub_loss_to_draw is None:
                        sub_loss_to_draw = torch.exp(-global_var)*global_err + trend_err
                    else:
                        sub_loss_to_draw += torch.exp(-global_var)*global_err + trend_err

                    local_loss_all[:, r_index_value - 140:r_index_value + 340 ,:] += torch.exp(-local_var)*local_err


                if loss_to_draw is None:
                    loss_to_draw = sub_loss_to_draw / 3
                else:
                    loss_to_draw += (sub_loss_to_draw / 3)

            else:
                continue
        
        loss_to_draw /= valid_rpeak_cnt
        loss_to_draw += (local_loss_all / 3)[:,100:4900,:]
        loss_to_draw = loss_to_draw.detach().cpu().numpy()

        for lds in range(12):
            loss_to_draw[0,:,lds] = gaussian_filter(loss_to_draw[0,:,lds], sigma=15)


        beat_division_result = np.array(beat_division_result)
        result.append(loss_to_draw)

    scores = np.asarray(result)
    auc_result = roc_auc_score(labels[:,100:4900,:].flatten(), scores.flatten())
    print(("Localization AUC: ", round(auc_result, 3)))
    return auc_result



if __name__ == '__main__':
    main()




