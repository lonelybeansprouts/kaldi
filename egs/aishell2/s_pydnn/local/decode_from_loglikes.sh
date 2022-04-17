#!/usr/bin/env bash

if [[ $# -ne 5 ]]; then
    echo "usage: [option] <trans_model_path> <graph_path> <loglikes_path> <output_path>"
fi

transcript
kaldi_root=/opt/kaldi
python_cmd=python3


. ./parse_options.sh

trans_model_path=$1
graph_path=$2
loglikes_path=$3
output_path=$4

${kaldi_root}/src/bin/decode-faster-mapped --word-symbol-table=${graph_path}/word.sym  \
                                          ${trans_model_path}/final.mdl     \
                                          ${graph_dir}/HCLG.fst     \
                                          "ark:${loglikes_path}/loglikes.ark"  \
                                          "ark,t:${output_path}/prediction.int" \
                                          > ${output_path}/prediction.txt

${python_cmd} word2char.py ${output_path}/prediction.txt > ${output_path}/prediction_chars.txt

if [[ ! -z transcript ]]; then
    ${kaldi_root}/src/bin/compute-wer --text --mode=present 
        ark:$transcript ark:${output_path}/prediction_chars.txt >${output_path}/wer
fi