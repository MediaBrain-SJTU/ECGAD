import time
import random
import numpy as np
import heartpy as hp
import torch
import copy
import torch.nn.functional as F
from scipy.signal import hilbert, cheby1, filtfilt

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



def generate_trend(ecg, device):

    avg_filter = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=10, stride=1, padding=0, groups=1, bias=False)
    avg_kernel = np.array([1/10]*10)
    avg_kernel = torch.from_numpy(avg_kernel).view(1,1,10).float().to(device)
    avg_filter.weight.data = avg_kernel
    avg_filter.weight.requires_grad = False


    dif_filter = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=4, stride=1, padding=0, groups=1, bias=False)
    dif_kernel = np.array([-1, 1])
    dif_kernel = torch.from_numpy(dif_kernel).view(1,1,2).float().to(device)
    dif_filter.weight.data = dif_kernel
    dif_filter.weight.requires_grad = False

    result = None
    # print("here ecg shape: ", ecg.shapes)
    for i in range(4):
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


# Forward and backward filtering using filtfilt.
def cheby1_bandpass_filter(data, lowcut, highcut, fs, order=5, rp=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = cheby1(order, rp=rp, Wn=[low, high], btype='bandpass')
    y = filtfilt(b, a, data)
    return y


# Running mean filter function from stackoverflow
# https://stackoverflow.com/a/27681394/6205282
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def FUNC_r_detection(EKG, sampling_rate = 500):
    EKG_f = cheby1_bandpass_filter(EKG, lowcut=6, highcut=18, fs=sampling_rate, order=4)
    dn = (np.append(EKG_f[1:], 0) - EKG_f)

    dtn = dn/(np.max(abs(dn)))
    an = abs(dtn)
    sn = -(dtn**2) * np.log10(dtn**2)

    window_len = 79
    sn_f = np.insert(running_mean(sn, window_len), 0, [0] * (window_len - 1))
    zn = np.imag(hilbert(sn_f))

    ma_len = 1250
    zn_ma = np.insert(running_mean(zn, ma_len), 0, [0] * (ma_len - 1))
    zn_ma_s = zn - zn_ma

    idx = np.argwhere(np.diff(np.sign(zn_ma_s)) > 0).flatten().tolist()
    idx_search = []
    id_maxes = np.empty(0, dtype=int)
    search_window_half = round(sampling_rate * .12)  # <------------ Parameter
    for i in idx:
        lows = np.arange(i-search_window_half, i)
        highs = np.arange(i+1, i+search_window_half+1)
        if highs[-1] > len(EKG)-10:
            break
            # highs = np.delete(highs, np.arange(np.where(highs == len(EKG))[0], len(highs)))
        ekg_window = np.concatenate((lows, [i], highs))
        idx_search.append(ekg_window)
        ekg_window_wave = EKG[ekg_window]
        id_maxes = np.append(id_maxes, ekg_window[np.where(ekg_window_wave == np.max(ekg_window_wave))[0]])
    
    final_idx = []
    for elem in id_maxes:
        if elem > 200 and elem < 4800-400:
            final_idx.append(elem)
    final_idx = np.array(final_idx)

    return final_idx