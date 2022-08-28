#!/usr/bin/env bash
# Copyright 2018 AIShell-Foundation(Authors:Jiayu DU, Xingyu NA, Bengu WU, Hao ZHENG)
#           2018 Beijing Shell Shell Tech. Co. Ltd. (Author: Hui BU)
#           2018 Emotech LTD (Author: Xuechen LIU)
# Apache 2.0

set -e

# number of jobs
nj=20
stage=5

. ./cmd.sh
[ -f ./path.sh ] && . ./path.sh;
. ./utils/parse_options.sh

# nj for dev and test
dev_nj=$(wc -l data/dev/spk2utt | awk '{print $1}' || exit 1;)
test_nj=$(wc -l data/test/spk2utt | awk '{print $1}' || exit 1;)

# Now make MFCC features.
if [ $stage -le 1 ]; then
  # mfccdir should be some place with a largish disk where you
  # want to store MFCC features.
  for x in train dev test; do
    steps/make_mfcc_pitch.sh --pitch-config conf/pitch.conf --cmd "$train_cmd" --nj $nj \
      data/$x exp/make_mfcc/$x mfcc || exit 1;
    steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x mfcc || exit 1;
    utils/fix_data_dir.sh data/$x || exit 1;
  done
fi

# mono
if [ $stage -le 2 ]; then
  # training
  steps/train_mono.sh --cmd "$train_cmd" --nj $nj \
    data/train data/lang exp/mono || exit 1;

  # decoding
  utils/mkgraph.sh data/lang_test exp/mono exp/mono/graph || exit 1;
  steps/decode.sh --cmd "$decode_cmd" --config conf/decode.conf --nj ${dev_nj} \
    exp/mono/graph data/dev exp/mono/decode_dev
  steps/decode.sh --cmd "$decode_cmd" --config conf/decode.conf --nj ${test_nj} \
    exp/mono/graph data/test exp/mono/decode_test
  
  # alignment
  steps/align_si.sh --cmd "$train_cmd" --nj $nj \
    data/train data/lang exp/mono exp/mono_ali || exit 1;
fi 

# tri1
if [ $stage -le 3 ]; then
  # training
  steps/train_deltas.sh --cmd "$train_cmd" \
   4000 32000 data/train data/lang exp/mono_ali exp/tri1 || exit 1;
  
  # decoding
  utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph || exit 1;
  steps/decode.sh --cmd "$decode_cmd" --config conf/decode.conf --nj ${dev_nj} \
    exp/tri1/graph data/dev exp/tri1/decode_dev
  steps/decode.sh --cmd "$decode_cmd" --config conf/decode.conf --nj ${test_nj} \
    exp/tri1/graph data/test exp/tri1/decode_test
  
  # alignment
  steps/align_si.sh --cmd "$train_cmd" --nj $nj \
    data/train data/lang exp/tri1 exp/tri1_ali || exit 1;
fi

# tri2
if [ $stage -le 4 ]; then
  # training
  steps/train_deltas.sh --cmd "$train_cmd" \
   7000 56000 data/train data/lang exp/tri1_ali exp/tri2 || exit 1;

  # decoding
  utils/mkgraph.sh data/lang_test exp/tri2 exp/tri2/graph
  steps/decode.sh --cmd "$decode_cmd" --config conf/decode.conf --nj ${dev_nj} \
    exp/tri2/graph data/dev exp/tri2/decode_dev
  steps/decode.sh --cmd "$decode_cmd" --config conf/decode.conf --nj ${test_nj} \
    exp/tri2/graph data/test exp/tri2/decode_test
  
  # alignment
  steps/align_si.sh --cmd "$train_cmd" --nj $nj \
    data/train data/lang exp/tri2 exp/tri2_ali || exit 1;
fi

# tri3
if [ $stage -le 5 ]; then
  # training [LDA+MLLT]
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
   10000 80000 data/train data/lang exp/tri2_ali exp/tri3 || exit 1;

  # decoding
  utils/mkgraph.sh data/lang_test exp/tri3 exp/tri3/graph || exit 1;
  steps/decode.sh --cmd "$decode_cmd" --nj ${dev_nj} --config conf/decode.conf \
    exp/tri3/graph data/dev exp/tri3/decode_dev
  steps/decode.sh --cmd "$decode_cmd" --nj ${test_nj} --config conf/decode.conf \
    exp/tri3/graph data/test exp/tri3/decode_test
  
  # alignment
  steps/align_si.sh --cmd "$train_cmd" --nj $nj \
    data/train data/lang exp/tri3 exp/tri3_ali || exit 1;
  
  steps/align_si.sh --cmd "$train_cmd" --nj 1 \
    data/dev data/lang exp/tri3 exp/tri3_ali_dev || exit 1;

  steps/align_si.sh --cmd "$train_cmd" --nj 1 \
    data/test data/lang exp/tri3 exp/tri3_ali_test || exit 1;
fi

#gen info for other usage
if [ $stage -le 6 ]; then
  local/gen_ali_txt.sh exp/tri3 exp/tri3_ali data/train  || exit 1;
  local/gen_ali_txt.sh exp/tri3 exp/tri3_ali_dev data/dev  || exit 1;
  local/gen_ali_txt.sh exp/tri3 exp/tri3_ali_test data/test  || exit 1;
  num_targets=$(tree-info exp/tri3/tree |grep num-pdfs|awk '{print $2}')
  echo $num_targets > exp/tri3/pdf_num
  cat exp/tri3/phones.txt | grep -v "#" | wc -l > exp/tri3/phone_num
  local-extract-pdfs-of-phone exp/tri3/final.mdl  `grep spn data/lang/phones.txt | awk '{print $2}'` | sort -u > exp/tri3/unk_pdfs.txt
fi

echo "local/run_gmm.sh succeeded"
exit 0;

