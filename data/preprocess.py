import csv
import pandas as pd
import os
import numpy as np
import re
import heartpy as hp
from collections import defaultdict
import copy
import wfdb
import ast



def natural_sort(input_list):
    def alphanum_key(key):
        return [int(s) if s.isdigit() else s.lower() for s in re.split("([0-9]+)", key)]
    return sorted(input_list, key=alphanum_key)


def normalize(X_train_ori):
    '''
    normalize to [-1,1]
    :param seq:
    :return:
    '''
    X_train = copy.deepcopy(X_train_ori)
    for count in range(X_train.shape[0]):
        for j in range(12):
            seq = X_train[count][:,j]
            X_train[count][:,j] = 2*(seq-np.min(seq))/(np.max(seq)-np.min(seq))-1
    return X_train


def hp_preprocess(X_train):
    train_data = []
    for i in range(X_train.shape[0]):
        ecg_tmp = []
        for lead in range(12):
            ecg = X_train[i][:,lead]

            filtered = hp.filter_signal(ecg,sample_rate=500, filtertype="highpass", cutoff=1)
            filt = hp.filter_signal(filtered, sample_rate=500, cutoff=35 ,filtertype="notch")
            y = hp.filter_signal(filt, sample_rate=500, filtertype="lowpass", cutoff=25)

            ecg_tmp.append(y)

        ecg_tmp = np.array(ecg_tmp).T
        train_data.append(ecg_tmp)

    train_data = np.array(train_data)
    
    return train_data



def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        # LR stands for lower rate, as this data is sampled at 100 Hz
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        # This is sampled at 500 Hz
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


def preprocess_ptbxl(path, sampling_rate = 500):

    # load and convert annotation data
    Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X = load_raw_data(Y, sampling_rate, path)

    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == True]

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    # Apply diagnostic superclass
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

    # Split data into train and test
    test_fold = 10
    # Train
    X_train = X[np.where(Y.strat_fold != test_fold)]
    y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
    # Test
    X_test = X[np.where(Y.strat_fold == test_fold)]
    y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass



    train_data = []
    train_label = []
    count = 0
    for item in y_train:
        try:
            if item[0] == 'NORM':
                train_data.append(X_train[count])
                train_label.append(1)
            count += 1
        except:
            count += 1
    train_data = np.asarray(train_data)
    np.save('train.npy',train_data)

    test_label = []
    test_data = []
    count = 0
    for item in y_test:
        try:
            if item[0] == 'NORM':
                test_label.append(0)
            else:
                test_label.append(1)
            test_data.append(X_test[count])
            count += 1
        except:
            count += 1

    test_label = np.asarray(test_label)
    test_data = np.asarray(test_data)

    np.save('test.npy',test_data)
    np.save('label.npy',test_label)



def denoise_train(train_data):
    denoised_data = hp_preprocess(train_data)
    denoised_data = normalize(denoised_data)

    train_data_4save = []
    for i in range(denoised_data.shape[0]):
        try:
            working_data, measures = hp.process(denoised_data[i, :, 1], 500.0)
        except:
            continue
        train_data_4save.append(denoised_data[i])
    train_data_4save = np.array(train_data_4save)

    np.save('train.npy',train_data_4save)



def denoise_test(test_data, test_label):
    denoised_data = hp_preprocess(test_data)
    # denoised_data = normalize(denoised_data)
    test_data_4save = []
    test_label_4save = []
    for i in range(denoised_data.shape[0]):
        try:
            hp.process(denoised_data[i, :, 1], 500.0)
        except:
            continue
        test_data_4save.append(denoised_data[i])
        test_label_4save.append(test_label[i])
    test_data_4save = np.array(test_data_4save)
    test_label_4save = np.array(test_label_4save)
    np.save('test.npy', test_data_4save)
    np.save('label.npy', test_label_4save)



if __name__ == '__main__':
    path = './ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/'
    preprocess_ptbxl(path)

    train_data = np.load("train.npy")
    test_data = np.load("test.npy")
    test_label = np.load("label.npy")

    print("denoise train")
    denoise_train(train_data)

    print("denoise test")
    denoise_test(test_data, test_label)

