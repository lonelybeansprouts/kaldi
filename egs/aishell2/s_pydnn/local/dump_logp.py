#!/usr/bin/env python3
# -*- coding: utf-8 -*-  




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
import torch.nn.functional as F
import kaldi_io


from pycore.data.dataset import CeFeatScpDataset as CeDataset
from pycore.data.dataloader import CeDataloader
from pycore.models import lstm
from pycore.models.model_init import *
from pycore.utils import get_logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp_dir")
    parser.add_argument("-config")
    parser.add_argument("-batch_size", default=32, type=int, help="Override the batch size in the config")
    parser.add_argument("-data_loader_threads", default=0, type=int, help="number of workers for data loading")
    parser.add_argument('-feat_dim', default=0, type=int, help='input dim of model')
    parser.add_argument('-label_size', default=0, type=int, help='output dim of model')
    parser.add_argument('-data_path', default='', type=str, help='alignment file path for training')
    parser.add_argument('-model_path', default='', type=str, help='model for infference')
    parser.add_argument('-dump_path', default='', type=str, help='path for dumpping')
    parser.add_argument('-log_path', default='/dev/null', type=str, help='path for dumpping')

    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.data_path:
        config["data_conf"]["data_path"]=args.data_path
    if args.feat_dim > 0:
       config["model_conf"]["input_size"]=args.feat_dim
    if args.label_size > 0:
       config["model_conf"]["output_size"]=args.label_size

    logger = get_logger(args.log_path)
    logger.info(args.log_path)

    logger.info("inferrence starts with config {}".format(json.dumps(config, sort_keys=True, indent=4)))

    trainset = CeDataset(config, train=False)
    dataloader = CeDataloader(trainset,
                                       batch_size=args.batch_size,
                                       num_workers=args.data_loader_threads, test_only=True)
    # ceate model
    model = load_model(config)

    if th.cuda.is_available():
        model.cuda()

    # set output path
    if args.dump_path == "-":
        output_file=sys.stdout.buffer
    else:
        output_file=kaldi_io.open_or_fd(args.dump_path,'wb')
    
    assert os.path.isfile(args.model_path), "ERROR: model file {} does not exit!".format(args.model_path)

    checkpoint = th.load(args.model_path)
    state_dict = checkpoint['model']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(state_dict)
    model.eval()
    for i, data in enumerate(dataloader):  
        key = data["utt_ids"][0]    #batch=1
        logger.info("dump lop for key: "+ key)
        feat = data["x"]
        x = feat.to(th.float32)
        if th.cuda.is_available():
            x = x.cuda()
        prediction = model(x)

        prediction = F.log_softmax(prediction, dim=-1) # check 

        if th.cuda.is_available():
          prediction = prediction.cpu()

        prediction = th.squeeze(prediction, 0) #discard batch dim which equals 1
        kaldi_io.write_mat(output_file, prediction.detach().numpy(), key=key)
    # close output file
    output_file.close()

if __name__ == '__main__':
    main()
