import numpy as np
import os
import heartpy as hp
import copy
import random
from utils import normalize, beat_normalize

class TrainSet:
    def __init__(self, folder):
        self.train_data = np.load(os.path.join(folder, 'train.npy'))

    def checkR(self, ecg):
        working_data, measures = hp.process(ecg, 500.0)
        peak_list = working_data['peaklist']
        return peak_list

    def __len__(self):
        return self.train_data.shape[0]

    def __getitem__(self, index):
        ecg_instance = self.train_data[index]
        r_index_list = self.checkR(ecg_instance[:,1])

        r_idx = random.choice(r_index_list)
        while r_idx < 200 or r_idx > 4800-400:
            r_idx = random.choice(r_index_list)

        beat = ecg_instance[r_idx-140:r_idx+340,:]
        beat = beat_normalize(beat)

        return beat, ecg_instance[100:4900,:]



class TestSet:
    def __init__(self, folder):
        self.train_data = np.load(os.path.join(folder, 'test.npy'))

    def __len__(self):
        return self.train_data.shape[0]

    def checkR(self, ecg):
        working_data, measures = hp.process(ecg, 500.0)
        peak_list = working_data['peaklist']
        return np.array(peak_list)

    def __getitem__(self, index):
        ecg_instance = self.train_data[index]
        r_index = self.checkR(ecg_instance[:,1])
        return r_index, ecg_instance



class PixelTestSet:
    def __init__(self, folder):
        self.train_data = np.load(os.path.join(folder, 'benchmark_data.npy'))
        self.train_data = normalize(self.train_data)

    def __len__(self):
        return self.train_data.shape[0]


    def checkR(self, ecg):
        working_data, measures = hp.process(ecg, 500.0)
        peak_list = working_data['peaklist']
        return np.array(peak_list)


    def __getitem__(self, index):
        ecg_instance = self.train_data[index]
        r_index = self.checkR(ecg_instance[:,1])
        return r_index, ecg_instance