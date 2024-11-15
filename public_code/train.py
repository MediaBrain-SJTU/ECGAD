import os
import random
import argparse
import time
import math
import torch
import numpy as np
from tqdm import tqdm
from utils import time_string, convert_secs2time, AverageMeter, generate_trend, normalize
from model_template.models.model import AD_Class
from dataloader import DataSet
from sklearn.metrics import roc_auc_score
import copy
import warnings
from losses import AsymmetricLoss

warnings.filterwarnings("ignore")
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def main():
    parser = argparse.ArgumentParser(description='ECG anomaly detection')
    parser.add_argument('--data_path', type=str, default='data/')
    parser.add_argument('--epochs', type=int, default=50, help='maximum training epochs')
    parser.add_argument('--dims', type=int, default=12, help='dimension of the input data')
    parser.add_argument('--save_model', type=int, default=1, help='0 for discard, 1 for save model')
    parser.add_argument('--save_path', type=str, default='ckpt/mymodel.pt')
    parser.add_argument('--mask_ratio', type=int, default=30, help='mask ratio for self-restoration')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate of others in SGD')
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

    dset = DataSet(folder=os.path.join(args.data_path, 'train.csv'))
    train_loader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, shuffle=True, **kwargs)

    dtset = DataSet(folder=os.path.join(args.data_path, 'test.csv'))
    test_loader = torch.utils.data.DataLoader(dtset, batch_size=1, shuffle=False, **kwargs)

    # load model
    model = AD_Class(enc_in=args.dims, hidden=args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters() , lr=args.lr, weight_decay=1e-5)


    # start training
    start_time = time.time()
    epoch_time = AverageMeter()

    old_auc_result = 0
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, args.lr, epoch, args)
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        print(' {:3d}/{:3d} ----- [{:s}] {:s}'.format(epoch, args.epochs, time_string(), need_time))
        epoch_time.update(time.time() - start_time)
        start_time = time.time()

        train(args, model, epoch, train_loader, optimizer)
        auc_result = test(args, model, epoch, test_loader)
        if auc_result > old_auc_result:
            old_auc_result = auc_result
            if args.save_model == 1:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, args.save_path)
    print("final best auc: ", old_auc_result)


def train(args, model, epoch, train_loader, optimizer):
    model.train()
    total_losses = AverageMeter()
    l = torch.nn.MSELoss(reduction='none')
    loss_fn = AsymmetricLoss(gamma_neg=4, gamma_pos=10, clip=0.05,  disable_torch_grad_focal_loss=True)
    for i, (local_ecg, global_ecg, attribute, target) in tqdm(enumerate(train_loader)):

        global_ecg = global_ecg.float().cuda()
        trend = generate_trend(global_ecg)
        mask_global = copy.deepcopy(global_ecg)

        local_ecg = local_ecg.float().cuda()
        mask_local = copy.deepcopy(local_ecg)

        bs, local_length, dim = local_ecg.shape
        global_length = global_ecg.shape[1]

        # add mask to global ecg
        mask = torch.zeros((bs,global_length,1), dtype=torch.bool).cuda()
        patch_length = global_length // 100
        for j in random.sample(range(0,100), args.mask_ratio):
            mask[:, j*patch_length:(j+1)*patch_length] = 1
        mask_global = torch.mul(mask_global, ~mask)

        # add mask to local instance
        cut_length = local_length * args.mask_ratio // 100
        cut_idx = random.randint(1, local_length-cut_length-2)
        mask_local[:, cut_idx:cut_idx+cut_length ,:] = 0

        (gen_global, global_var), (gen_local, local_var), gen_trend, gen_attr, prediction = model(mask_global, mask_local, trend)

        global_err = (gen_global - global_ecg) ** 2
        local_err = (gen_local - local_ecg) ** 2
        trend_err = (gen_trend - global_ecg) ** 2
        attr_err = l(gen_attr, attribute)

        l_global = torch.mean(torch.exp(-global_var)*global_err) + torch.mean(global_var)
        l_local = torch.mean(torch.exp(-local_var)*local_err) + torch.mean(local_var)
        l_trend = torch.mean(trend_err)
        l_attr = torch.mean(attr_err)
        l_class = loss_fn(prediction, target)

        final_loss = l_global + l_local + l_trend + l_attr + l_class
        loss = final_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_losses.update(final_loss.item(), bs)
        
    print(('Train Epoch: {} Total_Loss: {:.6f}'.format(epoch, total_losses.avg)))



def test(args, model, epoch, test_loader):
    torch.zero_grad = True
    model.eval()
    Sig = torch.Sigmoid()

    labels = []
    scores = []

    for i, (ecg_instance, beat_instance, attribute, target) in tqdm(enumerate(test_loader)):
        beat_instance = beat_instance.float().cuda()
        ecg_instance = ecg_instance.float().cuda()
        trend = generate_trend(ecg_instance)
        target = target.float().cuda()

        (gen_global, global_var), (gen_local, local_var), gen_trend, _, prediction = model(ecg_instance, beat_instance, trend)
        prediction = Sig(prediction)
        
        scores.append(prediction.detach().cpu())
        labels.append(target.detach().cpu())

    scores = torch.cat(scores).numpy()
    labels = torch.cat(labels).numpy()

    auc_result = roc_auc_score(labels, scores)
    print(("AUC: ", round(auc_result, 3)))
    return auc_result


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


if __name__ == '__main__':
    main()



