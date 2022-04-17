# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen, Di Wu)
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
import sys

import torch
import yaml
from torch.utils.data import DataLoader
import kaldi_io

from pynet.dataset.dataset import Dataset
from pynet.tasks import init_asr_model
from pynet.utils.checkpoint import load_checkpoint
from pynet.utils.file_utils import read_symbol_table, read_non_lang_symbols
from pynet.utils.config import override_config

def load_prior(path):
    if len(path)<=0:
        return None
    with open(path) as f:
        lines = f.readlines()
        assert len(lines) == 1
        prior = lines[0]
        prior = torch.FloatTensor([float(x) for x in prior.split() if (x!='[' and x!=']')])
        prior = prior / prior.sum()
        return prior.log()



def get_args():
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--prior_counts',
                        type=str,
                        default='')
    parser.add_argument('--prior_scale',
                        type=float,
                        default=1.0)
    parser.add_argument('--symbol_table',
                        required=True,
                        help='model unit symbol table for training')
    parser.add_argument("--non_lang_syms",
                        help="non-linguistic symbol file. One symbol per line.")
    parser.add_argument('--penalty',
                        type=float,
                        default=0.0,
                        help='length penalty')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='asr result file')
    parser.add_argument('--mode',
                        choices=[
                            'attention', 'ctc_greedy_search',
                            'ctc_prefix_beam_search', 'attention_rescoring'
                        ],
                        default='attention',
                        help='decoding mode')
    parser.add_argument('--decoding_chunk_size',
                        type=int,
                        default=-1,
                        help='''decoding chunk size,
                                <0: for decoding, use full chunk.
                                >0: for decoding, use fixed chunk size as set.
                                0: used for training, it's prohibited here''')
    parser.add_argument('--num_decoding_left_chunks',
                        type=int,
                        default=-1,
                        help='number of left chunks for decoding')
    parser.add_argument('--simulate_streaming',
                        action='store_true',
                        help='simulate streaming inference')
    parser.add_argument('--reverse_weight',
                        type=float,
                        default=0.0,
                        help='''right to left weight for attention rescoring
                                decode mode''')
    parser.add_argument('--bpe_model',
                        default=None,
                        type=str,
                        help='bpe model for english part')
    parser.add_argument('--override_config',
                        action='append',
                        default=[],
                        help="override yaml config")
    parser.add_argument('--connect_symbol',
                        default='',
                        type=str,
                        help='used to connect the output characters')
    parser.add_argument('--ignore_eos_prob',
                        default=True,
                        type=bool,
                        help="drop the eos class prob")


    args = parser.parse_args()
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='%(asctime)s %(levelname)s %(message)s')

    logging.info(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.mode in ['ctc_prefix_beam_search', 'attention_rescoring'
                     ] and args.batch_size > 1:
        logging.fatal(
            'decoding mode {} must be running with batch_size == 1'.format(
                args.mode))
        sys.exit(1)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    symbol_table = read_symbol_table(args.symbol_table)
    test_conf = copy.deepcopy(configs['dataset_conf'])

    test_conf['filter_conf']['max_length'] = 102400
    test_conf['filter_conf']['min_length'] = 0
    test_conf['filter_conf']['token_max_length'] = 102400
    test_conf['filter_conf']['token_min_length'] = 0
    test_conf['filter_conf']['max_output_input_ratio'] = 102400
    test_conf['filter_conf']['min_output_input_ratio'] = 0
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['spec_sub'] = False
    test_conf['shuffle'] = False
    test_conf['sort'] = False
    if 'fbank_conf' in test_conf:
        test_conf['fbank_conf']['dither'] = 0.0
    elif 'mfcc_conf' in test_conf:
        test_conf['mfcc_conf']['dither'] = 0.0
    test_conf['batch_conf']['batch_type'] = "static"
    test_conf['batch_conf']['batch_size'] = args.batch_size
    non_lang_syms = read_non_lang_symbols(args.non_lang_syms)

    test_dataset = Dataset(args.data_type,
                           args.test_data,
                           symbol_table,
                           test_conf,
                           args.bpe_model,
                           non_lang_syms,
                           partition=False)

    test_data_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)

    # Init asr model from configs
    model = init_asr_model(configs)
    log_prior = load_prior(args.prior_counts)
    load_checkpoint(model, args.checkpoint)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)
    model.eval()
    output_file=sys.stdout.buffer

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_data_loader):
            keys = batch['keys']
            assert (len(keys) == 1) #only support batch size == 1
            key = keys[0]
            logging.info("process wav key:" + key)

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                num_utts, batch_data = model.module.unpack_test_data(batch, device)
            else:
                num_utts, batch_data = model.unpack_test_data(batch, device)

            prediction = model.get_logprobs(*batch_data)
            if args.ignore_eos_prob:
                prediction = prediction[:,:,:-1] #ignore eos
            if use_cuda:
                prediction = prediction.cpu()

            prediction = torch.squeeze(prediction, 0) #discard batch dim which equals 1
            if log_prior != None:
                assert prediction.size(1) == log_prior.size(0) #assert classes equal
                prediction = prediction-args.prior_scale * log_prior #substract prior

            kaldi_io.write_mat(output_file, prediction.detach().numpy(), key=key)
            
    output_file.close()


if __name__ == '__main__':
    main()
