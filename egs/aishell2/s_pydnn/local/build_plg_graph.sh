#!/usr/bin/env bash

. local/parse_options.sh

if [[ -f ./path.sh ]]; then
  . ./path.sh
fi

if [[ $# -ne 2 ]]; then
    echo "usage: [option] <lang_test_dir>  <target_dir>"
    exit 1
fi

lang_dir=$1
dir=$2
tmpdir=${dir}/local
mkdir -p ${tmpdir}





# if [[ $# -ne 3 ]]; then
#     echo "usage: [option] <lang_test_dir> <dict_dir> <lang_arpa_path> <target_dir>"
#     exit 1
# fi

# lang_dir=$1
# dict_dir=$2
# lm=$3
# dir=$4
# tmpdir=${dir}/local
# mkdir -p ${tmpdir}


# cp ${dict_dir}/lexicon.txt ${tmpdir}




# perl -ape 's/(\S+\s+)(.+)/${1}1.0\t$2/;' < ${tmpdir}/lexicon.txt > ${tmpdir}/lexiconp.txt || exit 1;


# ndisambig=`add_lex_disambig.pl ${tmpdir}/lexiconp.txt ${tmpdir}/lexiconp_disambig.txt`
# ndisambig=$[$ndisambig+1];
# ( for n in `seq 0 $ndisambig`; do echo '#'$n; done ) > $tmpdir/disambig.list



#cat ${tmpdir}/disambig.list | awk '{print $2}' > ${tmpdir}/disambig.int


# cat ${tmpdir}/lexicon.txt | awk '
# {
#   for (i=2; i<=NF; i++){
#     A[$i]=1;
#   }
# }
# END {
#   for (i in A){
#     print i;
#   }
# }' > ${tmpdir}/phones.list

# cat ${lang_dir}/phones.txt | awk '{print $1}' | grep -v '#' > ${tmpdir}/phones.list


# cat ${tmpdir}/phones.list ${tmpdir}/disambig.list |\
#       awk '{print $1 " " (NR-1)}' > ${tmpdir}/units.txt




# cat ${tmpdir}/units.txt | grep '#' | awk '{print $2}' > ${tmpdir}/disambig.int



# cat ${tmpdir}/lexicon.txt | awk '
# {
#   A[$1]=1;
# }
# END {
#   for (i in A){
#     print i;
#   }
# }' > ${tmpdir}/word.list 



# cat ${tmpdir}/word.list  | awk '
#   BEGIN {
#     print "<eps> 0";
#   } 
#   {
#     printf("%s %d\n", $1, NR);
#   }
#   END {
#     printf("#0 %d\n", NR+1);
#   }'  > ${tmpdir}/words.txt || exit 1;


# token_disambig_symbol=`grep \#0 ${tmpdir}/units.txt | awk '{print $2}'`
# word_disambig_symbol=`grep \#0 ${tmpdir}/words.txt | awk '{print $2}'`




# make_lexicon_fst.pl --pron-probs ${tmpdir}/lexiconp_disambig.txt 0 "sil" '#'$ndisambig | \
#        fstcompile --isymbols=${tmpdir}/units.txt --osymbols=${tmpdir}/words.txt \
#        --keep_isymbols=false --keep_osymbols=false |   \
#        fstaddselfloops  "echo $token_disambig_symbol |" "echo $word_disambig_symbol |" | \
#        fstarcsort --sort_type=olabel > ${tmpdir}/L.fst || exit 1;




# python3 local/gen_lexicon_fst_with_loops.py ${tmpdir}/lexicon_disambig.txt "#0" "#0" > ${tmpdir}/L.txt
# cat ${tmpdir}/L.txt | fstcompile --isymbols=${tmpdir}/units.txt --osymbols=${tmpdir}/words.txt \
#        --keep_isymbols=false --keep_osymbols=false |   \
#        fstrmepsilon |fstarcsort --sort_type=olabel > ${tmpdir}/L.fst || exit 1;


# gunzip -c $lm | grep -v '<s> <s>' | \
#   grep -v '</s> <s>' | \
#   grep -v '</s> </s>' | \
#   arpa2fst - | fstprint | \
#   eps2disambig.pl | s2eps.pl | fstcompile --isymbols=${tmpdir}/words.txt \
#       --osymbols=${tmpdir}/words.txt  --keep_isymbols=false --keep_osymbols=false | \
#   fstrmepsilon | fstarcsort --sort_type=ilabel > ${tmpdir}/G.fst


# set +e
# fstisstochastic ${tmpdir}/G.fst
# set -e

# fsttablecompose ${tmpdir}/L.fst ${tmpdir}/G.fst | fstdeterminizestar --use-log=true  |
#   fstminimizeencoded | fstarcsort --sort_type=ilabel > ${tmpdir}/LG.fst || exit 1;


# # Compile the tokens into FST
# python3 local/gen_phone_fst_with_loops.py ${tmpdir}/units.txt | fstcompile --isymbols=${tmpdir}/units.txt --osymbols=${tmpdir}/units.txt \
#    --keep_isymbols=false --keep_osymbols=false | fstarcsort --sort_type=olabel > ${tmpdir}/T.fst || exit 1;


python3 local/gen_phone_fst_with_loops.py ${lang_dir}/phones.txt | fstcompile --isymbols=${lang_dir}/phones.txt --osymbols=${lang_dir}/phones.txt \
   --keep_isymbols=false --keep_osymbols=false | fstarcsort --sort_type=olabel > ${tmpdir}/T.fst || exit 1;

fsttablecompose ${lang_dir}/L_disambig.fst ${lang_dir}/G.fst | fstdeterminizestar --use-log=true  |
  fstminimizeencoded | fstarcsort --sort_type=ilabel > $tmpdir/LG.fst || exit 1;

fsttablecompose ${tmpdir}/T.fst $tmpdir/LG.fst > ${tmpdir}/TLG.fst  || exit 1;

cp ${lang_dir}/G.fst ${dir}/G.fst
cp ${lang_dir}/L.fst ${dir}/L.fst
cp ${tmpdir}/T.fst ${dir}/T.fst
cp ${tmpdir}/LG.fst ${dir}/LG.fst
cp ${tmpdir}/TLG.fst ${dir}/TLG.fst
cp ${lang_dir}/words.txt ${dir}/words.txt
cp ${lang_dir}/phones.txt ${dir}/phones.txt

rm -rf ${tmpdir}

# cp ${tmpdir}/G.fst ${dir}/G.fst
# cp ${tmpdir}/L.fst ${dir}/L.fst
# cp ${tmpdir}/T.fst ${dir}/T.fst
# cp ${tmpdir}/LG.fst ${dir}/LG.fst
# cp ${tmpdir}/TLG.fst ${dir}/TLG.fst
# cp ${tmpdir}/words.txt ${dir}/words.txt
# cp ${tmpdir}/units.txt ${dir}/units.txt



# python3 gen_lexicon_fst_with_loops.py ${tmpdir}/lexicon_disambig.txt  |
# fstcompile --isymbols=${tmpdir}/units.txt --osymbols=${tmpdir}/words.txt \
#        --keep_isymbols=false --keep_osymbols=false |   \
#        fstaddselfloops  "echo $token_disambig_symbol |" "echo $word_disambig_symbol |" | \
#        fstarcsort --sort_type=olabel > $dir/L.fst || exit 1;

