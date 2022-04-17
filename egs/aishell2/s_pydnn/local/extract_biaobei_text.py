#! /usr/bin/env python3

import yaml
import sys

data_info_path = sys.argv[1]
info = yaml.load(open(data_info_path), Loader=yaml.FullLoader)
for item in info:
    print(item['key'], ' ', item['text'])
