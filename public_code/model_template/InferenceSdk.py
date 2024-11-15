import os
import random
import argparse
import time
import pandas as pd
import torch
import numpy as np
from .models.utils import time_string, convert_secs2time, AverageMeter, generate_trend, normalize, FUNC_r_detection
from .models.model import AD_Class
import copy
import json
import pickle
import warnings
import heartpy as hp
warnings.filterwarnings("ignore")
from .data_preprocess import preprocess


class RatiocinationSdk:
    def __init__(self, gpu_id, weight_dir):

        self.gpuid = gpu_id
        if self.gpuid < 0:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(f'cuda:{gpu_id}')

        self.weight_dir = weight_dir

        self.model = AD_Class(enc_in=4, hidden=50).to(self.device)
        checkpoint = torch.load(os.path.join(weight_dir, 'weight.pt'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        torch.zero_grad = True
        self.model.eval()
        self.activate = torch.nn.Sigmoid()

        self.text_list = [
            "Normal electrocardiogram",
            "T wave changes",
            "Sinus bradycardia",
            "ST-T segment changes",
            "Sinus tachycardia",
            "ST segment changes",
            "Mild T wave changes",
            "First-degree atrioventricular block",
            "Left ventricular high voltage",
            "Atrial fibrillation",
            "Complete right bundle branch block",
            "Atrial premature beat",
            "Sinus arrhythmia",
            "Ventricular premature beat",
            "Low voltage",
            "Left anterior fascicular block",
            "Mild ST segment changes",
            "ST segment saddleback elevation",
            "Incomplete right bundle branch block",
            "ST segment depression",
            "Peaked T wave",
            "Clockwise rotation",
            "Atrial flutter",
            "Prominent U wave",
            "Complete left bundle branch block",
            "Ventricular paced rhythm",
            "Atrial tachycardia",
            "Intraventricular conduction block",
            "Left ventricular hypertrophy",
            "Paroxysmal atrial tachycardia",
            "Inferior myocardial infarction",
            "Mild ST-T segment changes",
            "Frequent atrial premature beat",
            "Extensive anterior myocardial infarction",
            "Anteroseptal Myocardial Infarction",
            "Frequent ventricular premature beat",
            "Abnormal Q wave",
            "Prolonged QT interval",
            "Right axis deviation",
            "Old anterior myocardial infarction",
            "Old inferior myocardial infarction",
            "Left axis deviation",
            "Atrial paced rhythm",
            "Pre-excitation syndrome",
            "Intraventricular conduction delay",
            "Flat T wave",
            "Poor R wave progression or reversed progression",
            "Non-conducted atrial premature beat",
            "Junctional escape beat",
            "Insertional ventricular premature beat",
            "Short PR interval with normal QRS complex",
            "Second-degree atrioventricular block",
            "Couplet atrial premature beat",
            "Atrial bigeminy",
            "Paroxysmal supraventricular tachycardia",
            "Elevated J point",
            "Paced electrocardiogram",
            "Peaked P wave",
            "Couplet ventricular premature beat",
            "Non-paroxysmal junctional tachycardia",
            "Biphasic P wave",
            "Third-degree atrioventricular block",
            "Second-degree type 1 atrioventricular block",
            "Horizontal ST segment depression",
            "Junctional premature beat",
            "Ventricular tachycardia",
            "Ventricular escape beat",
            "Anterior myocardial infarction",
            "Inverted T wave",
            "Posterior myocardial infarction",
            "Long RR interval",
            "Atrial trigeminy",
            "Paroxysmal ventricular tachycardia",
            "High lateral myocardial infarction",
            "Upsloping ST segment depression",
            "Downsloping ST segment depression",
            "Second-degree sinoatrial block",
            "High-degree atrioventricular block",
            "Right ventricular myocardial infarction",
            "Dextrocardia",
            "Right ventricular hypertrophy",
            "Ventricular bigeminy",
            "Second-degree type 2 sinoatrial block",
            "Second-degree type 1 sinoatrial block",
            "Atrial tachycardia with variable conduction",
            "Sinus arrest",
            "Biphasic P wave",
            "Lateral myocardial infarction",
            "Old lateral myocardial infarction",
            "Second-degree type 2 atrioventricular block",
            "Left posterior fascicular block",
            "Biphasic T wave",
            "Muscle artifact",
            "Old posterior myocardial infarction",
            "Inverted P wave",
            "Lead detachment",
            "Sinus node wandering rhythm",
            "Unstable baseline",
            "Ventricular trigeminy",
            "Subendocardial myocardial infarction",
            "Old high lateral myocardial infarction",
            "Atrial escape beat",
            "Atrial arrhythmia",
            "Broadened P wave",
            "Insertional atrial premature beat",
            "Interference atrioventricular dissociation",
            "Counterclockwise rotation",
            "Left atrial hypertrophy",
            "Right atrial hypertrophy",
            "Paroxysmal junctional tachycardia",
            "Ventricular quadrigeminy",
            "Ventricular fibrillation",
            "Shortened QT interval",
            "Alternating left and right bundle branch block",
            "Baseline drift",
            "Biauricular hypertrophy"
        ]


    def classify(self, input_json_path):

        ecg_instance, beat_instance = preprocess(input_json_path)
        beat_instance = beat_instance.float().to(self.device)
        ecg_instance = ecg_instance[:,100:4900,:].float().to(self.device)
        trend = generate_trend(ecg_instance, self.device)
        (gen_global, global_var), (gen_local, local_var), gen_trend, pred_attr, prediction = self.model(ecg_instance, beat_instance, trend)
        prediction = self.activate(prediction)
        prediction = torch.mean(prediction, dim=0)
        prediction = prediction.detach().cpu().numpy()
        

        all_disease = []
        confidence_idx = np.argsort(prediction)[::-1]
        for j in range(0,5):
            index = confidence_idx[j]
            all_disease.append(self.text_list[index])
        return all_disease

        




