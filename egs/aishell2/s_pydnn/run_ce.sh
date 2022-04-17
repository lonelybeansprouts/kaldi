#!/bin/bash


export CUDA_VISIBLE_DEVICES=0,1,2,3

stage=2
nj=32

python_cmd="python3 -u"


data_source=../s5/data
test_sets="dev test"
train_set=train

gmm_model_dir=../s5/exp/tri3
graph_dir=../s5/exp/tri3/graph
dict_dir=../s5/data/local/dict

train_align_source=../s5/exp/tri3_ali
train_ali_dir=data/train_alignment 


affix=pdf_dfsmn

# train_conf=conf/ce_lstm.yaml  
# decode_conf=conf/ce_lstm_decode.yaml  #%WER 10.52
train_conf=conf/dfsmn.yaml  
decode_conf=conf/dfsmn_decode.yaml  

mkdir -p exp/ce_$affix

echo "affix ${affix} start"


. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if [ $stage -le 0 ]; then
  fbankdir=data/fbank_hires
  for datadir in ${train_set} ${test_sets}; do
    utils/copy_data_dir.sh ${data_source}/${datadir} data/${datadir}_hires
    steps/make_fbank.sh --fbank-config conf/fbank_hires.conf --nj $nj data/${datadir}_hires exp/make_fbank/ ${fbankdir}
    utils/fix_data_dir.sh data/${datadir}_hires
  done
  compute-cmvn-stats scp:data/${train_set}_hires/feats.scp data/${train_set}_hires/cmvn.ark
  for datadir in ${train_set} ${test_sets}; do
    ${train_cmd} JOB=1:${nj} exp/make_fbank/apply_cmvn.JOB.log  \
        apply-cmvn --norm-vars=true data/${train_set}_hires/cmvn.ark scp:${fbankdir}/raw_fbank_${datadir}_hires.JOB.scp ark:- \| \
        copy-feats --compress=true --compression-method=2 ark:- ark,scp:${fbankdir}/cmvn_fbank_${datadir}_hires.JOB.ark,${fbankdir}/cmvn_fbank_${datadir}_hires.JOB.scp 
    cat ${fbankdir}/cmvn_fbank_${datadir}_hires.*.scp > data/${datadir}_hires/feats.scp
    utils/fix_data_dir.sh data/${datadir}_hires
  done

  cat ${graph_dir}/phones.txt | grep -v "#" > exp/ce_$affix/phones.txt
  echo "# $( cat exp/ce_$affix/phones.txt | wc -l )" >> exp/ce_$affix/phones.txt
  cat ${dict_dir}/lexicon.txt > exp/ce_$affix/lexicon.txt
  cat ${graph_dir}/words.txt > exp/ce_$affix/words.txt
fi



# prepare alignment
if [ $stage -le 1 ]; then
  local/gen_ali.sh --model $gmm_model_dir/final.mdl  \
                              $train_align_source \
                              $train_ali_dir  || exit 1;
  mv data/${train_set}_hires/feats.scp data/${train_set}_hires/feats.scp.bak
  utils/filter_scp.pl -f 1 data/${train_set}_hires/feats.scp.bak $train_ali_dir/pdfs.scp | sort -u -k 1  > data/${train_set}_hires/labels.scp
  utils/filter_scp.pl -f 1 data/${train_set}_hires/labels.scp data/${train_set}_hires/feats.scp.bak | sort -u -k 1  >  data/${train_set}_hires/feats.scp
fi


# train ce
if [ $stage -le 2 ]; then
  num_targets=$(tree-info $gmm_model_dir/tree |grep num-pdfs|awk '{print $2}')
  echo $num_targets > exp/ce_$affix/output_dim
  feat_dim=$(feat-to-dim scp:data/${train_set}_hires/feats.scp - || exit 1;)
  echo $feat_dim > exp/ce_$affix/feat_dim

  ${python_cmd} pycore/bin/train_ce.py -config $train_conf \
    -exp_dir exp/ce_$affix \
    -lr 0.0001 \
    -batch_size 64 \
    -data_loader_threads $nj \
    -anneal_lr_epoch 3 \
    -num_epochs 8 \
    -anneal_lr_ratio 0.5 \
    -feat_dim ${feat_dim} \
    -label_size ${num_targets} \
    -data_path data/${train_set}_hires
fi

exit 1


# decode for test
if [ $stage -le 3 ]; then
  for model_path in exp/ce_$affix/model.*.tar; do
    for test in $test_sets; do
        echo $model_path
        model_name=$( basename $model_path )
        decode_dir=exp/ce_$affix/${model_name//./_}_${test}_decode
        mkdir -p $decode_dir
        local/dump_logp.py   \
          -config $decode_conf \
          -feat_dim $( cat exp/ce_$affix/feat_dim ) \
          -label_size  $( cat exp/ce_$affix/output_dim )\
          -data_path  data/${test}_hires\
          -batch_size 1 \
          -model_path ${model_path} \
          -dump_path -  \
          -log_path ${decode_dir}/dump.log  |   \
        latgen-faster-mapped-parallel  \
          --num-threads=${nj} \
          ${gmm_model_dir}/final.mdl \
          ${graph_dir}/HCLG.fst \
          "ark:-" \
          "ark:|gzip -c >${decode_dir}/lat.1.gz" 
        steps/score_kaldi.sh data/${test}_hires ${graph_dir} ${decode_dir}
        steps/scoring/score_kaldi_cer.sh --stage 2 data/${test}_hires ${graph_dir} ${decode_dir} 
    done 
  done
fi

local/show_results.sh

echo "done!"

exit 0;


