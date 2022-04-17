#!/usr/bin/env bash

# This script is based on swbd/s5c/local/nnet3/run_tdnn.sh

# this is the standard "tdnn" system, built in nnet3; it's what we use to
# call multi-splice.

# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call train_tdnn.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.

# results
# local/nnet3/compare_wer.sh exp/nnet3/tdnn_sp/
# Model                  tdnn_sp
# WER(%)                    11.20
# Final train prob        -0.9601
# Final valid prob        -1.0819

set -e

stage=0
train_stage=-10
affix=
common_egs_dir=

# training options
initial_effective_lrate=0.0015
final_effective_lrate=0.00015
num_epochs=4
num_jobs_initial=2
num_jobs_final=6
nj=32
remove_egs=true

# feature options
use_ivectors=false

# End configuration section.

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi


dir=exp/nnet3/tdnn_sp${affix:+_$affix}
gmm_dir=exp/tri3
test_sets="dev test"
train_set=train
ali_dir=${gmm_dir}_ali
graph_dir=$gmm_dir/graph

if [ $stage -le 6 ]; then
  fbankdir=fbank_hires
  for datadir in ${train_set} ${test_sets}; do
    utils/copy_data_dir.sh data/${datadir} data/${datadir}_hires
    steps/make_fbank.sh --fbank-config conf/fbank.conf --nj $nj data/${datadir}_hires exp/make_fbank/ ${fbankdir}
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
fi


if [ $stage -le 7 ]; then
  echo "$0: creating neural net configs";
  affine_opts="l2-regularize=0.008 dropout-proportion=0.0 dropout-per-dim=true dropout-per-dim-continuous=true"
  tdnnf_opts="l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.75"
  linear_opts="l2-regularize=0.008 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.008"
  num_targets=$(tree-info $ali_dir/tree |grep num-pdfs|awk '{print $2}')
  feat_dim=$(feat-to-dim scp:data/${train_set}_hires/feats.scp - || exit 1;)

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=$feat_dim name=input
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2) affine-transform-file=$dir/configs/lda.mat
  relu-batchnorm-dropout-layer name=tdnn1 $affine_opts dim=1536
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=1
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=1
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=1
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=0
  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf14 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf15 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf16 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf17 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  linear-component name=prefinal-l dim=256 $linear_opts

  prefinal-layer name=prefinal input=prefinal-l $prefinal_opts big-dim=1536 small-dim=256
  output-layer name=output input=prefinal dim=$num_targets max-change=1.5
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi


if [ $stage -le 8 ]; then
  #if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
  #  utils/create_split_dir.pl \
  #   /export/b0{5,6,7,8}/$USER/kaldi-data/egs/aishell-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  #fi

  steps/nnet3/train_dnn.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
    --trainer.optimization.final-effective-lrate $final_effective_lrate \
    --egs.dir "$common_egs_dir" \
    --cleanup.remove-egs $remove_egs \
    --cleanup.preserve-model-interval 500 \
    --use-gpu true \
    --feat-dir=data/${train_set}_hires \
    --ali-dir $ali_dir \
    --lang data/lang \
    --reporting.email="$reporting_email" \
    --dir=$dir  || exit 1;
fi

if [ $stage -le 9 ]; then
  for decode_set in $test_sets; do
    # this version of the decoding treats each utterance separately
    # without carrying forward speaker information.
    num_jobs=`cat data/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
    decode_dir=${dir}/decode_$decode_set
    steps/nnet3/decode.sh --nj $num_jobs --cmd "$decode_cmd" \
       $graph_dir data/${decode_set}_hires $decode_dir || exit 1;
  done
fi

wait;
echo "local/nnet3/run_tdnn.sh succeeded"
exit 0;
