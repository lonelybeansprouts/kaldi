# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import copy
import logging
import os

import torch
import torch.distributed as dist
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from pynet.dataset.dataset import Dataset
from pynet.utils.file_utils import read_symbol_table, read_non_lang_symbols
from pynet.utils.config import override_config

def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--data', required=True, help='train data file')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--cmvn', default=None, help='global cmvn file')
    parser.add_argument('--symbol_table',
                        required=True,
                        help='model unit symbol table for training')
    parser.add_argument("--non_lang_syms",
                        help="non-linguistic symbol file. One symbol per line.")
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='prefetch number')
    parser.add_argument('--bpe_model',
                        default=None,
                        type=str,
                        help='bpe model for english part')

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')

    # Set random seed
    torch.manual_seed(777)
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)


    symbol_table = read_symbol_table(args.symbol_table)

    dataset_conf = configs['dataset_conf']

    non_lang_syms = read_non_lang_symbols(args.non_lang_syms)

    dataset = Dataset(args.data_type, args.data, symbol_table,
                            dataset_conf, args.bpe_model, non_lang_syms, True)

    dataloader = DataLoader(dataset,
                            batch_size=None,
                            pin_memory=args.pin_memory,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch)

    i=0
    while True:
        logging.info("batch data for iter: " + str(i))
        for batch_data in dataloader:
            i = i+1
            print(batch_data)
            if (i==1):
                break
        break



if __name__ == '__main__':

### usage example 
# python3 -u pynet/bin/data_test.py  \
#         --config $train_config \
#         --data_type raw \
#         --symbol_table $token_table \
#         --data data/$train_set/data.list \
#         --num_workers 2

    main()
