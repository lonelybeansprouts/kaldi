#!/usr/bin/env bash

# Copyright 2018 AIShell-Foundation(Authors:Jiayu DU, Xingyu NA, Bengu WU, Hao ZHENG)
#           2018 Beijing Shell Shell Tech. Co. Ltd. (Author: Hui BU)
#           2018 Emotech LTD (Author: Xuechen LIU)
# Apache 2.0



stage=1

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if [ $# -ne 1 ]; then
  echo "prepare_all.sh <corpus-train-dir> <corpus-dev-dir> <corpus-test-dir>"
  echo " e.g prepare_all.sh /data/AISHELL-2/iOS/train /data/AISHELL-2/iOS/dev /data/AISHELL-2/iOS/test"
  exit 1;
fi

download_data=$1
data_url=www.openslr.org/resources/33


# download and extract data 
if [ $stage -le 1 ]; then
  local/download_and_untar.sh $download_data $data_url data_aishell || exit 1;
  local/download_and_untar.sh $download_data $data_url resource_aishell || exit 1;
fi

# download DaCiDian raw resources, convert to Kaldi lexicon format
if [ $stage -le 2 ]; then
  local/prepare_dict_BigCiDian.sh data/local/dict || exit 1;
fi

# prepare data
if [ $stage -le 3 ]; then
  local/aishell_data_prep.sh $download_data/data_aishell/wav $download_data/data_aishell/transcript data/local/dict || exit 1;
fi

# L
if [ $stage -le 4 ]; then
  utils/prepare_lang.sh --position-dependent-phones false \
    data/local/dict "<UNK>" data/local/lang data/lang || exit 1;
fi


# arpa LM
if [ $stage -le 5 ]; then
  local/train_lms.sh \
      data/local/dict/lexicon.txt \
      data/local/train/text \
      data/local/lm || exit 1;
fi

# G compilation, check LG composition
if [ $stage -le 6 ]; then
  utils/format_lm.sh data/lang data/local/lm/3gram-mincount/lm_unpruned.gz \
    data/local/dict/lexicon.txt data/lang_test || exit 1;
fi

echo "local/prepare_all.sh succeeded"
exit 0;

