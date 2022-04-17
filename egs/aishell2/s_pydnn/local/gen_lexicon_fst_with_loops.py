#!/usr/bin/env python

# Apache 2.0

import sys

if (len(sys.argv)-1==1) :
    fread = open(sys.argv[1], 'r')  
elif (len(sys.argv)-1==3) :
    fread = open(sys.argv[1], 'r')
    #处理语言模型回退概率的消歧符
    input_disambig=sys.argv[2]
    output_disambig=sys.argv[3]
else:
    print("usages: gen_lexicon_fst_with_loops.py lexicon.txt [#0 #0]")




print('0 1 <eps> <eps>')
print('2 0 <eps> <eps>')

nodeX = 3
for entry in fread.readlines():
    fields = entry.replace('\n','').strip().split()
    assert len(fields) > 1
    word = fields[0]
    phone = fields[1]
    if '#' in phone:
      print(str(1) + ' ' + str(nodeX) + ' ' +  phone + ' <eps>')
    else:
      print(str(1) + ' ' + str(nodeX) + ' ' +  phone + ' <eps>')
     # print(str(nodeX) + ' ' + str(nodeX) + ' ' +  phone + ' <eps>')
    next_phone=phone
    for next_phone in fields[2:]:
      if '#' in next_phone:
        print(str(nodeX) + ' ' + str(nodeX+1) + ' ' +  next_phone + ' <eps>')
      else:
        print(str(nodeX) + ' ' + str(nodeX+1) + ' ' +  next_phone + ' <eps>')
       # print(str(nodeX+1) + ' ' + str(nodeX+1) + ' ' +  next_phone + ' <eps>')
      nodeX += 1
    #print(str(nodeX) + ' ' + str(nodeX) + ' ' + input_disambig+ ' ' + output_disambig) #生成回退概率的消歧符
    print(str(nodeX) + ' ' + str(2) + ' ' + '<eps>' + ' ' + word)
    nodeX += 1
print('0')

fread.close()
