#!/usr/bin/env python3

from pycore.kaldi_io import kaldi_io
import sys
import numpy as np


if (len(sys.argv) != 4):
    print("usage example: local/insert_phone_boundary.py  ali.ark phone.txt <boundary_id>")
    sys.exit(1)


ali_ark_path = sys.argv[1]
phone_map_path = sys.argv[2]
boundary_phone_id = int(sys.argv[3])

phone_map = {}
for line in open(phone_map_path, 'r').readlines():
    key, value = line.split()
    phone_map[key] = value


sil_phone_id = int(phone_map["sil"])


for key, value in kaldi_io.read_vec_int_ark(ali_ark_path):
    value = value.tolist()
    size = len(value)

    phone_count = 0
    ignore = False
    for i in range(size):
        if value[i] == sil_phone_id:
            continue
        phone_count = phone_count+1
        if (i==size-1 or value[i]!=value[i+1]):
            if (phone_count == 1):
                ignore = True
                break
            value[i] = boundary_phone_id
            phone_count = 0

    value = np.array(value, dtype=np.int32)

    if not ignore:
        kaldi_io.write_vec_int(sys.stdout.buffer, v=value, key=str(key))