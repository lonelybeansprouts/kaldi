"""
Copyright (c) 2019 Microsoft Corporation. All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import yaml
import argparse
import numpy as np
import os
import sys
import time
import json
import pickle

import torch as th
import torch.nn as nn

from pycore.data.dataset import CeFeatScpDataset as CeDataset
from pycore.data.dataloader import CeDataloader
from pycore.utils import utils
from pycore.models.model_init import load_model

import warpctc_pytorch as warp_ctc
import pickle



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp_dir")
    parser.add_argument("-config")
    parser.add_argument("-lr", default=0.001, type=float, help="Override the LR in the config")
    parser.add_argument("-batch_size", default=32, type=int, help="Override the batch size in the config")
    parser.add_argument("-data_loader_threads", default=0, type=int, help="number of workers for data loading")
    parser.add_argument("-max_grad_norm", default=5, type=float, help="max_grad_norm for gradient clipping")
    parser.add_argument("-num_epochs", default=1, type=int, help="number of training epochs (default:1)")
    parser.add_argument("-resume_from_model", type=str, help="the model from which you want to resume training")
    parser.add_argument("-dropout", default=0.5, type=float, help="set the dropout ratio")
    parser.add_argument("-anneal_lr_epoch", default=2, type=int, help="start to anneal the learning rate from this epoch") 
    parser.add_argument("-anneal_lr_ratio", default=0.5, type=float, help="the ratio to anneal the learning rate")
    parser.add_argument('-print_freq', default=100, type=int, metavar='N', help='print frequency (default: 100)')
    parser.add_argument('-feat_dim', default=0, type=int, help='input dim of model')
    parser.add_argument('-label_size', default=0, type=int, help='output dim of model')
    parser.add_argument('-data_path', default='', type=str, help='alignment file path for training')
    parser.add_argument("-ctc_weight", default=1.0, type=float, help="ctc weight for training")


    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.data_path:
        config["data_conf"]["data_path"]=args.data_path
    if args.feat_dim > 0:
       config["model_conf"]["input_size"]=args.feat_dim
    if args.label_size > 0:
       config["model_conf"]["output_size"]=args.label_size


    print("Experiment starts with config {}".format(json.dumps(config, sort_keys=True, indent=4)))
    if not os.path.isdir(args.exp_dir):
        os.makedirs(args.exp_dir)

    trainset = CeDataset(config, train=True)

    dataloader = CeDataloader(trainset,
                                       batch_size=10,
                                       num_workers=args.data_loader_threads)
    
    # ceate model
    model = load_model(config)

    # optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)

    # criterion
    criterion_ce = nn.CrossEntropyLoss(ignore_index=-100)
    criterion_bfctc = warp_ctc.BFCTCLoss(length_average=True)


    start_epoch = 0
    if args.resume_from_model:
        assert os.path.isfile(args.resume_from_model), "ERROR: model file {} does not exit!".format(args.resume_from_model)
        checkpoint = th.load(args.resume_from_model)
        state_dict = checkpoint['model']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' ".format(args.resume_from_model))
    

    for epoch in range(start_epoch, args.num_epochs):
         # aneal learning rate
        if epoch > args.anneal_lr_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.anneal_lr_ratio

        run_train_epoch(model, optimizer, criterion_ce, criterion_bfctc, dataloader, epoch, args)


def run_train_epoch(model, optimizer, criterion_ce, criterion_bfctc, dataloader, epoch, args):
    model.train()
    device = th.device("cpu")
    if th.cuda.is_available():
        th.backends.cudnn.enabled = True
        device = th.device("cuda")
    if th.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print("use gpus: {}".format(th.cuda.device_count()))
    model.to(device)


    batch_time = utils.AverageMeter('time', ':6.3f')
    ce_losses = utils.AverageMeter('ce_Loss', ':.4e')
    grad_norm = utils.AverageMeter('grad_norm', ':.4e')
    meters = [batch_time, ce_losses, grad_norm]
    if (args.ctc_weight > 0.0):
        ctc_losses = utils.AverageMeter('ctc_Loss', ':.4e')
        meters.append(ctc_losses)


    progress = utils.ProgressMeter(len(dataloader), batch_time, *meters, grad_norm,
                             prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    # trainloader is an iterator. This line extract one minibatch at one time
    for i, data in enumerate(dataloader, 0):
        feat = data["x"]
        label = data["y"]

        x = feat.to(th.float32)
        y = label.unsqueeze(2).long()

        if th.cuda.is_available():
            x = x.to(device)
            y = y.to(device)

        prediction = model(x)

        ce_loss = (1-args.ctc_weight) * criterion_ce(prediction.view(-1, prediction.shape[2]), y.view(-1))

        # update ce loss
        ce_losses.update(ce_loss.item(), x.size(0))

        loss = ce_loss

        if (args.ctc_weight > 0.0):#run on cpu
            prediction = prediction.transpose(0, 1) #transpose to (T, batch, feat_dim) and must be before softmax
            prediction = prediction.cpu()
            labels = data['ctc_labels']
            prob_lens = data['num_frs']
            label_lens = data['ctc_label_lens']

            # pickle.dump([prediction, labels, prob_lens, label_lens], open("data.bin", 'wb'))

            ctc_loss = criterion_bfctc(prediction, labels, prob_lens, label_lens)

            # sys.exit(0)

            # update ctc loss
            ctc_losses.update(ctc_loss[0].item(), x.size(0))

            loss += args.ctc_weight * ctc_loss[0].to(device)

        optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping
        norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        #update grad norm
        grad_norm.update(norm)


        # measure elapsed time
        batch_time.update(time.time() - end)

        if i % args.print_freq == 0:
            progress.print(i)
    
    # save model
    if th.cuda.device_count() > 1:
        model = model.module
    checkpoint={}
    checkpoint['model']=model.state_dict()
    checkpoint['optimizer']=optimizer.state_dict()
    checkpoint['epoch']=epoch
    output_file=args.exp_dir + '/model.'+ str(epoch) +'.tar'
    th.save(checkpoint, output_file)


if __name__ == '__main__':
    main()
