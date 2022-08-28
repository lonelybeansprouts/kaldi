#!/bin/bash


export CUDA_VISIBLE_DEVICES=0,1,2,3

# The num of machines(nodes) for multi-machine training, 1 is for one machine.
# NFS is required if num_nodes > 1.
num_nodes=1

# The rank of each node or machine, which ranges from 0 to `num_nodes - 1`.
# You should set the node_rank=0 on the first machine, set the node_rank=1
# on the second machine, and so on.
node_rank=0
# The aishell dataset location, please change this to your own path
# make sure of using absolute path. DO-NOT-USE relatvie path!

export OMP_NUM_THREADS=8  #set to batch_size

stage=4
nj=32

python_cmd="python3 -u"

train_set=train
dev_set=dev
test_set=test

gmm_model_dir=exp/tri3
graph_dir=exp/tri3/graph
dict_dir=data/local/dict
ali_dir=exp/tri3_ali


affix=transformer_pdf_bfctc_multiattention
dir=exp/py_$affix
mkdir -p $dir

train_config=conf/pyconf/train_unified_transformer_bfctc_multiattention.yaml
token_table=$dir/dict/lang_char.txt
data_type=raw

checkpoint=
average_checkpoint=true
decode_checkpoint=$dir/final.pt
average_num=10


echo "affix ${affix} start"


. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if [ $stage -le 1 ]; then
    # remove the space between the text labels for Mandarin dataset
    for x in ${train_set} ${dev_set} ${test_set}; do
        paste -d " " <(cat data/${x}/text | sed -r 's%\t% %g' | cut -d " " -f 1 ) \
            <(cat data/${x}/text | sed -r 's%\t% %g' | cut -d " " -f 2- \
            | tr 'a-z' 'A-Z' | sed 's/\([A-Z]\) \([A-Z]\)/\1▁\2/g' | tr -d " ") \
            > data/${x}/text.tgt
    done
    pynet/tools/compute_cmvn_stats.py --num_workers ${nj} --train_config $train_config \
        --in_scp data/${train_set}/wav.scp \
        --out_cmvn data/$train_set/global_cmvn
fi

if [ $stage -le 2 ]; then
    echo "Make a dictionary"
    mkdir -p $(dirname $token_table)
    echo "<blank> 0" > ${token_table} # 0 will be used for "blank" in CTC
    echo "<unk> 1" >> ${token_table} # <unk> must be 1
    pynet/tools/text2token.py -s 1 -n 1 data/${train_set}/text.tgt | cut -f 2- -d" " | tr " " "\n" \
        | sort | uniq | grep -a -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${token_table}
    num_token=$(cat $token_table | wc -l)
    echo "<sos/eos> $num_token" >> $token_table # <eos>
fi


if [ $stage -le 3 ]; then
    # Prepare wenet requried data
    echo "Prepare data, prepare requried format"
    for x in ${train_set} ${dev_set}; do
        pynet/tools/make_raw_list.py --pdf_ali data/$x/pdfs.ali \
                                    --phone_ali data/$x/phones.ali \
                                    data/$x/wav.scp data/$x/text.tgt data/$x/data.list
    done
    for x in ${test_set}; do
        pynet/tools/make_raw_list.py data/$x/wav.scp data/$x/text.tgt data/$x/data.list
    done
fi


if [[ $stage -le 4 ]]; then
  # You have to rm `INIT_FILE` manually when you resume or restart a
  # multi-machine training.
  INIT_FILE=$dir/ddp_init
  init_method=file://$(readlink -f $INIT_FILE)
  echo "$0: init method is $init_method"
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="gloo"
  world_size=`expr $num_gpus \* $num_nodes`
  echo "total gpus is: $world_size"
  cmvn_opts=
  $cmvn && cp data/${train_set}/global_cmvn $dir
  $cmvn && cmvn_opts="--cmvn ${dir}/global_cmvn"

  # train.py rewrite $train_config to $dir/train.yaml with model input
  # and output dimension, and $dir/train.yaml will be used for inference
  # and export.
  for ((i = 0; i < $num_gpus; ++i)); do
  {
    gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
    # Rank of each gpu/process used for knowing whether it is
    # the master of a worker.
    rank=`expr $node_rank \* $num_gpus + $i`
    python3 pynet/bin/train.py --gpu $gpu_id \
      --config $train_config \
      --data_type $data_type \
      --symbol_table $token_table \
      --pdf_num $gmm_model_dir/pdf_num \
      --phone_num $gmm_model_dir/phone_num \
      --train_data data/$train_set/data.list \
      --cv_data data/dev/data.list \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir $dir \
      --ddp.init_method $init_method \
      --ddp.world_size $world_size \
      --ddp.rank $rank \
      --ddp.dist_backend $dist_backend \
      --num_workers 1 \
      $cmvn_opts 
  } &
  done
  wait
fi


if [[ $stage -le 5 ]]; then
  # Test model, please specify the model you want to test by --checkpoint
  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$dir/avg_${average_num}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python3 pynet/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path $dir  \
      --num ${average_num} \
      --val_best
  fi
fi


# decode for test
if [ $stage -le 6 ]; then
    model_path=$decode_checkpoint
    dataset=test
    model_name=$( basename $model_path )
    decode_dir=$dir/${model_name//./_}_${dataset}_decode
    mkdir -p $decode_dir
    python3 -u pynet/bin/dump_logp.py --gpu -1 \
        --config $dir/train.yaml \
        --data_type $data_type \
        --symbol_table $token_table \
        --test_data data/test/data.list \
        --checkpoint $decode_checkpoint \
        --prior_counts data/train/pdf.prior.counts \
        --unk_pdfs $gmm_model_dir/unk_pdfs.txt \
        --unk_delta 15.0 \
        --batch_size 1  |
    latgen-faster-mapped-parallel  --acoustic-scale=0.1 \
        --word-symbol-table=data/lang_test/words.txt \
        --num-threads=${nj} \
        ${gmm_model_dir}/final.mdl \
        ${graph_dir}/HCLG.fst \
        "ark:-" \
        "ark:|gzip -c >${decode_dir}/lat.1.gz" 
    steps/score_kaldi.sh data/test ${graph_dir} ${decode_dir}
    steps/scoring/score_kaldi_cer.sh --stage 2 data/test ${graph_dir} ${decode_dir} 
fi

local/show_results.sh