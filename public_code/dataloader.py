import numpy as np
import os
import heartpy as hp
import copy
import random
from utils import normalize, beat_normalize
import pandas as pd
import torch
import json


class DataSet:
    def __init__(self, folder):
        self.frame = pd.read_csv(folder)

    def checkR(self, ecg):
        working_data, measures = hp.process(ecg, 500.0)
        peak_list = working_data['peaklist']
        return peak_list

    def normalize(self, seq):
        normalized_seq = 2*(seq-np.min(seq)) / (np.max(seq) - np.min(seq))
        return normalized_seq
    
    def __len__(self):
        return len(self.frame)


    def __getitem__(self, index):
        file_name = self.frame['file_path'][index]
        abnormal_flag = eval(self.frame['abnorm_flag'][index])
        attribute = eval(self.frame['attribute'][index])

        target = torch.zeros(116)
        for idx in abnormal_flag:
            target[idx] = 1

        with open(file_name, 'r', encoding='utf8') as fp:
            ecg_instance = json.load(fp)


        r_index_list = self.checkR(ecg_instance[:,1])
        r_idx = random.choice(r_index_list)
        beat = ecg_instance[r_idx-140:r_idx+340,:]

        return ecg_instance, beat, attribute, target


