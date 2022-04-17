

import sys
import torch.utils.data as data
import numpy as np
import os
import kaldi_io


#used for ce training
class CeFeatScpDataset(data.Dataset):
    def __init__(self, config, train=False, trans_cmd=None):  #type support: train, dev, test
        self.train = train
        self.trans_cmd = trans_cmd
        self.feats_data={}
        for line in open(config["data_conf"]["data_path"]+"/feats.scp").readlines():
            key, value = line.strip().split()
            self.feats_data[key] = value
            
        if self.train:
            self.align_data={}
            for line in open(config["data_conf"]["data_path"]+"/labels.scp").readlines():
               key, value = line.strip().split()
               self.align_data[key]=value
            assert(len(self.feats_data) == len(self.align_data))
        self.keys = list(self.feats_data.keys())
        self.len_ = len(self.keys)

    def __getitem__(self, index):
        item=list()
        key = self.keys[index]
        item.append(key)
        if self.trans_cmd:
            cmd = 'echo {} {} ' .format(key, self.feats_data[key])  \
                + self.trans_cmd + '| copy-feats ark:- ark:- 2>/dev/null'
        else:
            cmd = 'echo {} {} ' .format(key, self.feats_data[key])  \
                + '| copy-feats scp:- ark:- 2>/dev/null'
        #print(cmd)
        fd = kaldi_io.popen(cmd)
        _,feat = kaldi_io.read_mat_ark(fd).__next__()
        item.append(feat)
        fd.close()

        if self.train:
            fd = kaldi_io.open_or_fd(self.align_data[key])
            alignment = kaldi_io.read_vec_int(fd)
            fd.close()
            item.append(alignment.reshape(-1,1))
        return item

    def __len__(self):
        return self.len_






# #used for ce training
# class CeRawScpDataset(data.Dataset):
#     #type support: train, dev, test]
#     #trans_cmd: must end of ark:-
#     def __init__(self, config, dataset_type, trans_cmd=None):  
#         self.type = dataset_type
#         self.trans_cmd = trans_cmd
#         if self.type == "train":
#             self.data_path = config["data_config"]["train"]["data_path"]
#             self.align_path = config["data_config"]["train"]["align_path"]
#         elif self.type == "dev":
#             self.data_path = config["data_config"]["dev"]["data_path"]
#         elif self.type == "test":
#             self.data_path = config["data_config"]["test"]["data_path"]
#         else:
#             print("not support data type: "+ str(type))
#             exit(1)

#         self.raw_data={}
#         for line in open(self.data_path).readlines():
#             key, value = line.strip().split()
#             self.raw_data[key] = value
            
#         if self.type == "train":
#             self.align_data={}
#             for line in open(self.align_path).readlines():
#                key, value = line.strip().split()
#                self.align_data[key]=value
#             assert(len(self.raw_data) == len(self.align_data))
#         self.keys = list(self.raw_data.keys())
#         self.len_ = len(self.keys)


#     def __getitem__(self, index):
#         item=list()
#         key = self.keys[index]
#         item.append(key)
#         cmd = 'echo {} {} | python3 data/wav_enhancement.py | wav-copy ark:- ark:- |'.format(key, self.raw_data[key]) \
#              + self.trans_cmd
#         fd = kaldi_io.popen(cmd)
#         _, feat = kaldi_io.read_mat_ark(fd)
#         item.append(feat)
#         fd.close()
#         if self.type == "train":
#             fd = kaldi_io.open_or_fd(self.align_data[key])
#             alignment = kaldi_io.read_vec_int(fd)
#             fd.close()
#             item.append(alignment.reshape(-1,1))
#         return item

#     def __len__(self):
#         return self.len_


