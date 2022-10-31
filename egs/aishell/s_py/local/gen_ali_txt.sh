#!/usr/bin/env bash



. ./cmd.sh
[ -f ./path.sh ] && . ./path.sh;
. ./utils/parse_options.sh


if [[ $# -ne 3 ]]; then
    echo "usage: [option] <src_dir> <target_dir>"
    exit 1
fi

model_dir=$1
src_dir=$2
target_dir=$3

mkdir -p $target_dir/tmp

for x in `find $src_dir -name "ali.*.gz"`; do
    name=`basename $x`
    ark_name=${name%.*}.ark
    scp_name=${name%.*}.scp
    copy-int-vector "ark:gunzip -c $x|" "ark,scp:${target_dir}/tmp/${ark_name},${target_dir}/tmp/${scp_name}"
done

cat ${target_dir}/tmp/*.scp > $target_dir/tmp/ali.scp


ali-to-pdf $model_dir/final.mdl "scp:$target_dir/tmp/ali.scp" "ark,t:${target_dir}/pdfs.ali"
analyze-counts ark,t:${target_dir}/pdfs.ali ${target_dir}/pdf.prior.counts


ali-to-phones --per-frame $model_dir/final.mdl "scp:$target_dir/tmp/ali.scp" "ark,t:${target_dir}/phones.ali" 
analyze-counts ark,t:${target_dir}/phones.ali ${target_dir}/phone.prior.counts

rm -rf $target_dir/tmp