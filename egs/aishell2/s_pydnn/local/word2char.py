#!/usr/bin/python3
# encoding=utf-8
# Copyright 2018 AIShell-Foundation(Authors:Jiayu DU, Xingyu NA, Bengu WU, Hao ZHENG)
#           2018 Beijing Shell Shell Tech. Co. Ltd. (Author: Hui BU)
# Apache 2.0

from __future__ import print_function
import sys
import re


if len(sys.argv) < 2:
  sys.stderr.write("word_segmentation.py <vocab> <trans> > <word-segmented-trans>\n")
  exit(1)


trans_file=sys.argv[1]
#jieba.set_dictionary(vocab_file)
for line in open(trans_file):
  line = ' '.join(line.strip().split()) #将所有分割符换成空格
  key=None
  trans=None
  key,trans=line.strip().split(' ',1)
  words = trans.strip().split()
  tr_words = []
  for word in words:
    if (re.match("[a-zA-Z0-9]+", word, flags=0) != None):
      tr_words.append(word)
      continue
    for ch in word:
      tr_words.append(ch)
  
  out = ' '.join(tr_words)

  new_line = key + '\t' + out

  print(new_line)
