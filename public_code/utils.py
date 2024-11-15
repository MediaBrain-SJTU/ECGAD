import time
import random
import numpy as np
import heartpy as hp
import torch
import copy
import torch.nn.functional as F

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime(ISOTIMEFORMAT, time.localtime()))
    return string


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs


def normalize(X_train_ori):
    X_train = copy.deepcopy(X_train_ori)
    for count in range(X_train.shape[0]):
        for j in range(12):
            seq = X_train[count][:,j]
            X_train[count][:,j] = 2*(seq-seq.min())/(seq.max()-seq.min())-1
    return X_train


def beat_normalize(X_train_ori):
    X_train = copy.deepcopy(X_train_ori)
    for j in range(12):
        seq = X_train[:,j]
        X_train[:,j] = 2*(seq-np.min(seq))/(np.max(seq)-np.min(seq))-1
    return X_train



def generate_trend(ecg):

    avg_filter = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=10, stride=1, padding=0, groups=1, bias=False)
    avg_kernel = np.array([1/10]*10)
    avg_kernel = torch.from_numpy(avg_kernel).view(1,1,10).float().cuda()
    avg_filter.weight.data = avg_kernel
    avg_filter.weight.requires_grad = False


    dif_filter = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=4, stride=1, padding=0, groups=1, bias=False)
    dif_kernel = np.array([-1, 1])
    dif_kernel = torch.from_numpy(dif_kernel).view(1,1,2).float().cuda()
    dif_filter.weight.data = dif_kernel
    dif_filter.weight.requires_grad = False

    result = None
    for i in range(12):
        mit_row_dif = avg_filter(ecg[:,:,i:i+1].transpose(1,-1))
        mit_row_dif = dif_filter(mit_row_dif)
        tmp_result = F.pad(input=mit_row_dif, pad=(5, 5), mode='constant', value=0)
        if result is None:
            result = tmp_result.transpose(1,-1)
        else:
            tmp_result = tmp_result.transpose(1,-1)
            result = torch.cat([result, tmp_result], dim=-1)

    for count in range(result.shape[0]):
        seq = result[count]
        result[count] = 2*(seq-seq.min())/(seq.max()-seq.min())-1

    return result