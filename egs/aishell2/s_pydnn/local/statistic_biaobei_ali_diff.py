#! /usr/bin/env python3

import yaml
import sys
import numpy

prediction_info_path = sys.argv[1]
with open(prediction_info_path) as f:
    prediction_info = f.readlines()

prediction_info = [x.split(' ', 2) for x in prediction_info]
rtfs = [float(x[1].split("_")[-1]) for x in prediction_info]

prediction_map = {}
for item in prediction_info:
    key = item[0]
    raw_info = item[2]
    value = []
    start_pos = -1
    for i in range(len(raw_info)):
        if (raw_info[i] == '('):
            start_pos = i
        if (raw_info[i] == ')'):
            assert(raw_info[start_pos] == '(')
            item_info = raw_info[start_pos+1:i]
            item_info = item_info.split(' ', 2)
            value.append((str(item_info[0]), float(item_info[1]), float(item_info[2])))
        prediction_map[key] = value


grouandtruth_info_path = sys.argv[2]
info = yaml.load(open(grouandtruth_info_path), Loader=yaml.FullLoader)
total_start_diffs = []
total_end_diffs = []
for item in info:
    key = item['key']
    if (key=='000610' or key=="000611"):
        continue

    print("process key: ", key)
    predict_ali = prediction_map[key]
    alignments = item['interval_text']
    ali_start_diffs = []
    ali_end_diffs = []
    pos = 0
    print(alignments)
    print(predict_ali)
    for ali in alignments:
        word = ali[2]
        start_time = ali[0]
        end_time = ali[1]

        if (word=='sil' or word=='sp1'):
            continue

        predict_word = predict_ali[pos][0]
        predict_start_time = predict_ali[pos][1]
        predict_end_time = predict_ali[pos][2]

        if (predict_word == '<UNK>'):
            pos = pos+1
            continue

        if (word == predict_word):
            if (abs(start_time-predict_start_time)>1 or abs(end_time-predict_end_time)>1):
                assert False
            ali_start_diffs.append(abs(start_time-predict_start_time))
            ali_end_diffs.append(abs(end_time-predict_end_time))
            pos = pos+1
        else:
            print("word:", word)
            print("predict_word:", predict_word)
            assert False
    total_start_diffs.extend(ali_start_diffs)
    total_end_diffs.extend(ali_end_diffs)

total_start_diffs = numpy.array(total_start_diffs, dtype=numpy.float32)
total_end_diffs = numpy.array(total_end_diffs, dtype=numpy.float32)

print("ave rtf: ", sum(rtfs)/len(rtfs))
print("ave_start_diff: ", total_start_diffs.mean())
print("max_start_diff: ", total_start_diffs.max())
print("start_diff < 40: ", numpy.sum(total_start_diffs<0.04)/total_start_diffs.size)
print("start_diff < 80: ", numpy.sum(total_start_diffs<0.08)/total_start_diffs.size)
print("ave_end_diff: ", total_end_diffs.mean())
print("max_end_diff: ", total_end_diffs.max())
print("end_diff < 40: ", numpy.sum(total_end_diffs<0.04)/total_end_diffs.size)
print("end_diff < 80: ", numpy.sum(total_end_diffs<0.08)/total_end_diffs.size)