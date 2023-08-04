import numpy as np
import torch
import torch.nn as nn
import os
import heartpy as hp
from scipy.fft import fft, fftfreq, ifft
import scipy.signal as ss
from scipy.signal import kaiserord, firwin, filtfilt, butter
import copy
import torch.nn.functional as F
import math
import random


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        x = torch.matmul(attn, value)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2



class Encoder(nn.Module):
    def __init__(self, nc, out_z):
        super(Encoder, self).__init__()
        ndf=32
        self.main = nn.Sequential(
            nn.Conv1d(nc,ndf,4,2,1,bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ndf * 16, out_z, 15, 1, 0, bias=False),
        )

    def forward(self, input):
        output = self.main(input.transpose(-1,1))
        return output

class Decoder(nn.Module):
    def __init__(self, nc, out_z):
        super(Decoder, self).__init__()
        ngf = 32
        self.main=nn.Sequential(
            nn.ConvTranspose1d(out_z,ngf*16,15,1,0,bias=False),
            nn.BatchNorm1d(ngf*16),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf * 4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf*2),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf * 2, ngf , 4, 2, 1, bias=False),
            nn.BatchNorm1d(ngf ),
            nn.ReLU(True),
            nn.ConvTranspose1d(ngf , nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output


class MCF(nn.Module):

    def __init__(self, enc_in, hidden = 50):
        super(MCF, self).__init__()

        self.channel = enc_in

        self.global_encoder = Encoder(enc_in,hidden)
        self.global_decoder = Decoder(enc_in+1,hidden)

        self.local_encoder = Encoder(enc_in,hidden)
        self.local_decoder = Decoder(enc_in+1,hidden)

        self.trend_encoder = Encoder(enc_in,hidden)
        self.trend_decoder = Decoder(enc_in,hidden*2)

        
        self.local_mlp = nn.Sequential(
            nn.Linear(137, 137*2),
            nn.ReLU(True),
            nn.Linear(137*2, 1),
        )

        self.global_mlp = nn.Sequential(
            nn.Linear(137, 137*4),
            nn.ReLU(True),
            nn.Linear(137*4, 136),
        )

        self.attn = MultiHeadedAttention(2, hidden)
        self.drop = nn.Dropout(0.1)
        self.layer_norm = LayerNorm(hidden)

    def forward(self, global_ecg, local_ecg, trend):

        latent_global = self.global_encoder(global_ecg)
        latent_local = self.local_encoder(local_ecg)
        latent_trend = self.trend_encoder(trend)


        latent_combine = torch.cat([latent_global, latent_local], dim=-1)

        # attention block
        latent_combine = latent_combine.transpose(-1, 1)
        attn_latent = self.attn(latent_combine, latent_combine, latent_combine)
        attn_latent = self.layer_norm(latent_combine + self.drop(attn_latent))
        latent_combine = attn_latent.transpose(-1, 1)

        latent_local = latent_local + self.local_mlp(latent_combine)
        latent_global = latent_global + self.global_mlp(latent_combine)

        trend_combine = torch.cat([latent_global, latent_trend], dim=1)

        gen_global = self.global_decoder(latent_global)
        gen_global = gen_global.transpose(-1, 1)

        gen_local = self.local_decoder(latent_local)
        gen_local = gen_local.transpose(-1, 1)

        gen_trend = self.trend_decoder(trend_combine)
        gen_trend = gen_trend.transpose(-1, 1)


        return  (gen_global[:,:,0:self.channel],gen_global[:,:,self.channel:self.channel+1]), \
                (gen_local[:,:,0:self.channel], gen_local[:,:,self.channel:self.channel+1]), \
                gen_trend

