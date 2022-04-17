#!/usr/bin/env bash



model=final.mdl


. ./cmd.sh
[ -f ./path.sh ] && . ./path.sh;
. ./utils/parse_options.sh


if [[ $# -ne 2 ]]; then
    echo "usage: [option] <src_dir> <target_dir>"
    exit 1
fi

src_dir=$1
target_dir=$2

mkdir -p $target_dir/tmp

for x in `find $src_dir -name "ali.*.gz"`; do
    name=`basename $x`
    ark_name=${name%.*}.ark
    scp_name=${name%.*}.scp
    copy-int-vector "ark:gunzip -c $x|" "ark,scp:${target_dir}/tmp/${ark_name},${target_dir}/tmp/${scp_name}"
done

cat ${target_dir}/tmp/*.scp > ${target_dir}/ali.scp
copy-int-vector "scp:${target_dir}/ali.scp" "ark:${target_dir}/ali.ark"

ali-to-pdf $model "scp:${target_dir}/ali.scp" "ark,scp:${target_dir}/pdfs.ark,${target_dir}/pdfs.scp"

ali-to-phones --per-frame $model "scp:${target_dir}/ali.scp" "ark,scp:${target_dir}/phones.ark,${target_dir}/phones.scp"  

rm -rf $target_dir/tmp