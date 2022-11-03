#!/usr/bin/env bash

# Copyright 2018 AIShell-Foundation(Authors:Jiayu DU, Xingyu NA, Bengu WU, Hao ZHENG)
#           2018 Beijing Shell Shell Tech. Co. Ltd. (Author: Hui BU)
# Apache 2.0

# AISHELL-2 provides:
#  * a Mandarin speech corpus (~1000hrs), free for non-commercial research/education use
#  * a baseline recipe setup for large scale Mandarin ASR system
# For more details, read $KALDI_ROOT/egs/aishell2/README.txt

# modify this to your AISHELL-2 training data path
# e.g:
# trn_set=/disk10/data/AISHELL-2/iOS/data
# dev_set=/disk10/data/AISHELL-2/iOS/dev
# tst_set=/disk10/data/AISHELL-2/iOS/test

download_data=download_data

nj=32

stage=3
gmm_stage=0

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

# prepare trn/dev/tst data, lexicon, lang etc
if [ $stage -le 1 ]; then
  local/prepare_all.sh --stage 0 $download_data || exit 1;
fi

# GMM
if [ $stage -le 2 ]; then
  local/run_gmm.sh --nj $nj --stage $gmm_stage
fi


# DNN
if [ $stage -le 3 ]; then
  local/nnet3/run_tdnn_small.sh --nj $nj --stage 0 --affix small --num_jobs_initial 1 --num_jobs_final 1 --num_epochs 6
fi
exit 1


# chain
if [ $stage -le 4 ]; then
  local/chain/run_tdnn.sh --nj $njq
fi

local/show_results.sh

exit 0;
