#!/usr/bin/env bash
# Copyright 2018 AIShell-Foundation(Authors:Jiayu DU, Xingyu NA, Bengu WU, Hao ZHENG)
#           2018 Beijing Shell Shell Tech. Co. Ltd. (Author: Hui BU)
# Apache 2.0

# transform raw AISHELL-2 data to kaldi format

. ./path.sh || exit 1;

tmp=
dir=

if [ $# != 3 ]; then
  echo "Usage: $0 <corpus-data-dir> <dict_dir> <output-dir>"
  echo " $0 biaobei_tts data/biaobei_tts"
  exit 1;
fi

corpus_data_dir=$1
dict_dir=$2
dir=$3
tmp=$dir/tmp

echo "prepare_data.sh: Preparing data in $corpus"

mkdir -p $dir
mkdir -p $tmp


# corpus check
if [[ ! -d ${corpus_data_dir}/wav ]] || [[ ! -f ${corpus_data_dir}/data_info.yaml ]]; then
  echo "Error: $0 requires wav.scp and trans.txt under $corpus directory."
  exit 1;
fi

find ${corpus_data_dir} -name "*.wav" | sed -r 's%(.*)/(.+)\.wav%\2 &%' | sort > $tmp/tmp_wav.scp

local/extract_biaobei_text.py ${corpus_data_dir}/data_info.yaml | sort > $tmp/tmp_text.scp


awk '{print $1}' $tmp/tmp_wav.scp   > $tmp/wav_utt.list
awk '{print $1}' $tmp/tmp_text.scp > $tmp/trans_utt.list

utils/filter_scp.pl -f 1 $tmp/wav_utt.list $tmp/trans_utt.list > $tmp/utt.list
utils/filter_scp.pl -f 1 $tmp/utt.list $tmp/tmp_wav.scp | sort -k 1 > $tmp/wav.scp

# text
python -c "import jieba" 2>/dev/null || \
  (echo "jieba is not found. Use tools/extra/install_jieba.sh to install it." && exit 1;)
utils/filter_scp.pl -f 1 $tmp/utt.list $tmp/tmp_text.scp | sort -k 1 > $tmp/trans.txt
# jieba's vocab format requires word count(frequency), set to 99
awk '{print $1}' $dict_dir/lexicon.txt | sort | uniq | awk '{print $1,99}'> $tmp/word_seg_vocab.txt
python local/word_segmentation.py $tmp/word_seg_vocab.txt $tmp/trans.txt > $tmp/text

cat $tmp/utt.list | awk '{printf("%s %s\n", $1, $1)}' > $tmp/tmp_utt2spk

utils/filter_scp.pl -f 1 $tmp/utt.list $tmp/tmp_utt2spk | sort -k 1 | uniq > $tmp/utt2spk
utils/utt2spk_to_spk2utt.pl $tmp/utt2spk | sort -k 1 | uniq > $tmp/spk2utt

# copy prepared resources from tmp_dir to target dir
mkdir -p $dir
for f in wav.scp text spk2utt utt2spk; do
  cp $tmp/$f $dir/$f || exit 1;
done

rm -rf $tmp

echo "local/prepare_data.sh succeeded"
exit 0;
