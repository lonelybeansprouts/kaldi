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
affix=dfsmn_kaldi_infer


dir=exp/ali_$affix

mkdir -p $dir

echo "affix $dir start"

infer_sets="biaobei"

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if [ $stage -le 0 ]; then
  local/prepare_biaobei.sh data/download/biaobei_tts ${dict_dir} data/biaobei_hires
fi


if [ $stage -le 1 ]; then
  fbankdir=data/fbank_hires
  for datadir in ${infer_sets}; do
    steps/make_fbank.sh --fbank-config conf/fbank_hires.conf --nj $nj data/${datadir}_hires exp/make_fbank/ ${fbankdir}
    utils/fix_data_dir.sh data/${datadir}_hires
  done
  for datadir in ${infer_sets}; do
    ${train_cmd} JOB=1:${nj} exp/make_fbank/apply_cmvn.JOB.log  \
        apply-cmvn --norm-vars=true ${model_dir}/cmvn.ark scp:${fbankdir}/raw_fbank_${datadir}_hires.JOB.scp ark:- \| \
        copy-feats --compress=true --compression-method=2 ark:- ark,scp:${fbankdir}/cmvn_fbank_${datadir}_hires.JOB.ark,${fbankdir}/cmvn_fbank_${datadir}_hires.JOB.scp 
    cat ${fbankdir}/cmvn_fbank_${datadir}_hires.*.scp > data/${datadir}_hires/feats.scp
    utils/fix_data_dir.sh data/${datadir}_hires
  done
fi


# if [ $stage -le 3 ]; then
#   for datadir in ${infer_sets}; do
#     oov=`cat ${lang_dir}/oov.int`
#     local/split_char.py data/${datadir}_hires/text " " |
#             utils/sym2int.pl --map-oov $oov -f 2- ${model_dir}/words.txt - > data/${datadir}_hires/text.int

#     model_path=${model_dir}/${model}
#     echo $model_path
#     model_name=$( basename $model_path )
#     decode_dir=$dir/${model_name//./_}_${datadir}_ali_infer
#     mkdir -p $decode_dir
#     local/dump_logp.py   \
#         -config $decode_conf \
#         -feat_dim $( cat ${model_dir}/feat_dim ) \
#         -label_size  $( cat ${model_dir}/output_dim )\
#         -data_path  data/${datadir}_hires \
#         -batch_size 1 \
#         -model_path ${model_path} \
#         -dump_path -  \
#         -log_path ${decode_dir}/dump.log  |   \
#     local-fast-kaldi-align \
#         --read-disambig-syms=${lang_dir}/phones/disambig.int \
#         --word-symbol-table=${model_dir}/words.txt \
#         ${gmm_model_dir}/tree \
#         ${gmm_model_dir}/final.mdl  \
#         ${lang_dir}/L.fst  \
#         "data/${datadir}_hires/text.int" \
#         "ark:-" \
#         ${lang_dir}/phones/align_lexicon.int \
#         ${decode_dir}/ali_output
#   done
# fi

# local/statistic_biaobei_ali_diff.py ${decode_dir}/ali_output data/download/biaobei_tts/data_info.yaml

echo "done!"

exit 0;


