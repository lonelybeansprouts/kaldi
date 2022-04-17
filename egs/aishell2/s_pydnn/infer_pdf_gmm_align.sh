#!/bin/bash

export LANG=C.UTF-8

export CUDA_VISIBLE_DEVICES=0,1,2,3

stage=2
nj=32

python_cmd="python3 -u"

gmm_model_dir=../s5/exp/tri3
graph_dir=../s5/exp/tri3/graph
dict_dir=../s5/data/local/dict
lang_dir=../s5/data/lang_test

# decode_conf=conf/ce_lstm_decode.yaml 
decode_conf=conf/dfsmn_decode.yaml  
# model_dir=exp/ce_pdf_blstm
model_dir=exp/ce_pdf_dfsmn
model=model.7.tar
# affix=lstm_kaldi_infer
affix=gmm_infer


dir=exp/ali_$affix

mkdir -p $dir

echo "affix $dir start"

infer_sets="biaobei"

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if [ $stage -le 0 ]; then
  local/prepare_biaobei.sh data/download/biaobei_tts ${dict_dir} data/biaobei
fi

if [ $stage -le 1 ]; then
  mfcc=data/mfcc
  for datadir in ${infer_sets}; do
    steps/make_mfcc_pitch.sh --pitch-config conf/pitch.conf --nj $nj data/${datadir} exp/make_mfcc ${mfcc}
    steps/compute_cmvn_stats.sh data/${datadir} exp/make_mfcc ${mfcc} || exit 1;
    utils/fix_data_dir.sh data/${datadir}
  done
fi




if [ $stage -le 2 ]; then
  for datadir in ${infer_sets}; do
    feats='ark,s,cs:apply-cmvn --utt2spk=ark:data/${datadir}/utt2spk scp:data/${datadir}/cmvn.scp scp:data/${datadir}/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats ${gmm_model_dir}/final.mat ark:- ark:- |'
    oov=`cat ${lang_dir}/oov.int`
    local/split_char.py data/${datadir}/text " " |
            utils/sym2int.pl --map-oov $oov -f 2- ${model_dir}/words.txt - > data/${datadir}/text.int

    decode_dir=$dir/$(basename ${gmm_model_dir})_${datadir}_ali_infer
    mkdir -p $decode_dir   

    apply-cmvn --utt2spk=ark:data/${datadir}/utt2spk scp:data/${datadir}/cmvn.scp scp:data/${datadir}/feats.scp ark:- | \
    splice-feats ark:- ark:- | \
    transform-feats ${gmm_model_dir}/final.mat ark:- ark:- | \
    gmm-compute-likes   ${gmm_model_dir}/final.mdl ark:- ark:-  |   \
    local-fast-kaldi-align \
        --read-disambig-syms=${lang_dir}/phones/disambig.int \
        --word-symbol-table=${model_dir}/words.txt \
        ${gmm_model_dir}/tree \
        ${gmm_model_dir}/final.mdl  \
        ${lang_dir}/L.fst  \
        "data/${datadir}/text.int" \
        "ark:-" \
        ${lang_dir}/phones/align_lexicon.int \
        ${decode_dir}/ali_output
  done
fi


# local/statistic_biaobei_ali_diff.py ${decode_dir}/ali_output data/download/biaobei_tts/data_info.yaml



echo "done!"

exit 0;


