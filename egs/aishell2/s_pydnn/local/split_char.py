#! /usr/bin/env python3
import sys

data_info_path = sys.argv[1]
d = sys.argv[2]
with open(data_info_path) as f:
    for x in f.readlines():
        key, info = x.split(d, 1)
        info = [x for x in info]
        print(key, " ", " ".join(info))
        

