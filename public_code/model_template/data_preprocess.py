
import os
import random
import time
import pandas as pd
import torch
import numpy as np
import json
import pickle
from .models.utils import time_string, convert_secs2time, AverageMeter, generate_trend, normalize, FUNC_r_detection
import heartpy as hp


def normalize(seq):
    normalized_seq = 2*(seq - np.min(seq))/(np.max(seq)-np.min(seq))-1
    return normalized_seq

def denoise(current_lead):
    first_part = current_lead[::2]
    second_part = current_lead[1::2]

    first_part = hp.filter_signal(first_part,sample_rate=500, filtertype="highpass", cutoff=1)
    first_part = hp.filter_signal(first_part, sample_rate=500, cutoff=35 ,filtertype="notch")
    first_part = hp.filter_signal(first_part, sample_rate=500, filtertype="lowpass", cutoff=25)
    
    second_part = hp.filter_signal(second_part,sample_rate=500, filtertype="highpass", cutoff=1)
    second_part = hp.filter_signal(second_part, sample_rate=500, cutoff=35 ,filtertype="notch")
    second_part = hp.filter_signal(second_part, sample_rate=500, filtertype="lowpass", cutoff=25)

    first_part = normalize(first_part)
    second_part = normalize(second_part)
    return first_part, second_part

def preprocess(file_name):

    with open(file_name, 'r', encoding='utf-8') as file:
        data = json.load(file)
        
    ecg_signal_first = []
    ecg_signal_second = []

    for lead in ['I', 'II', 'III']:
        current_lead_first, current_lead_second = denoise(data[lead])
        ecg_signal_first.append(current_lead_first[0:1250])
        ecg_signal_second.append(current_lead_second[0:1250])

    for lead in ['aVR', 'aVL', 'aVF']:
        current_lead_first, current_lead_second = denoise(data[lead])
        ecg_signal_first.append(current_lead_first[1250:2500])
        ecg_signal_second.append(current_lead_second[1250:2500])

    for lead in ['V1', 'V2', 'V3']:
        current_lead_first, current_lead_second = denoise(data[lead])
        ecg_signal_first.append(current_lead_first[2500:3750])
        ecg_signal_second.append(current_lead_second[2500:3750])

    for lead in ['V4', 'V5', 'V6']:
        current_lead_first, current_lead_second = denoise(data[lead])
        ecg_signal_first.append(current_lead_first[3750:5000])
        ecg_signal_second.append(current_lead_second[3750:5000])


    ecg_signal_first = np.asarray(ecg_signal_first).T
    ecg_signal_second = np.asarray(ecg_signal_second).T


    combine_ecg_first = ecg_signal_first[:, 0:3]
    combine_ecg_second = ecg_signal_second[:, 0:3]
    for tmpcnt in [3,6,9]:
        combine_ecg_first = np.concatenate([combine_ecg_first, ecg_signal_first[:, tmpcnt:tmpcnt+3]], axis=0)
        combine_ecg_second = np.concatenate([combine_ecg_second, ecg_signal_second[:, tmpcnt:tmpcnt+3]], axis=0)


    # long_ecg_first = np.expand_dims(denoise((data['II'][:5000])), axis=-1)
    # long_ecg_second = np.expand_dims(denoise((data['II'][5000:])), axis=-1)
    long_ecg_first, long_ecg_second = denoise(data['II'])
    long_ecg_first = np.asarray([long_ecg_first]).T
    long_ecg_second = np.asarray([long_ecg_second]).T

    combine_ecg_first = np.concatenate([combine_ecg_first, long_ecg_first], axis=-1)
    combine_ecg_second = np.concatenate([combine_ecg_second, long_ecg_second], axis=-1)


    combine_ecg_first = torch.tensor(combine_ecg_first).unsqueeze(0)
    combine_ecg_second = torch.tensor(combine_ecg_second).unsqueeze(0)

    r_index_list = FUNC_r_detection(long_ecg_first[:,0])

    if len(r_index_list) == 0:
        r_idx = random.randint(201, 4400)
    else:
        r_idx = random.choice(r_index_list)
    beat_first = combine_ecg_first[:,r_idx-140:r_idx+340,:]

    r_index_list = FUNC_r_detection(long_ecg_second[:,0])
    if len(r_index_list) == 0:
        r_idx = random.randint(201, 4400)
    else:
        r_idx = random.choice(r_index_list)
    beat_second = combine_ecg_second[:,r_idx-140:r_idx+340,:]

    combine_ecg = torch.cat([combine_ecg_first, combine_ecg_second])
    beat = torch.cat([beat_first, beat_second])

    # combine_ecg = torch.tensor(combine_ecg).unsqueeze(0)
    # beat = torch.tensor(beat).unsqueeze(0)

    return combine_ecg, beat