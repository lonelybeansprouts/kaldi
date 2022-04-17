"""
Copyright (c) 2019 Microsoft Corporation. All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import numpy as np
import operator
import torch
from itertools import groupby

from torch.utils.data import Dataset, DataLoader
      
class CeDataloader(DataLoader):
    
    def __init__(self, dataset, batch_size, num_workers=0, test_only=False):
        
        self.test_only = test_only
 
        # now decide on a sampler
        #base_sampler = torch.utils.data.SequentialSampler(self.dataset)
        base_sampler = torch.utils.data.RandomSampler(dataset)
        sampler = torch.utils.data.BatchSampler(base_sampler, batch_size, False)
        super(CeDataloader, self).__init__(dataset,
                                        batch_sampler=sampler,
                                        num_workers=num_workers,
                                        collate_fn=self.collate_fn)
   

    def collate_fn(self, batch):
        
        def pad_and_concat_feats(inputs):
            # max_t = max(inp.shape[0] for inp in inputs)
            num_frs = [inp.shape[0] for inp in inputs]
            max_t = max(num_frs)
            shape = (len(inputs), max_t, inputs[0].shape[1])
            input_mat = np.zeros(shape, dtype=np.float32)
            for e, inp in enumerate(inputs):
                input_mat[e, :inp.shape[0], :] = inp
            return input_mat, num_frs
        
        def pad_and_concat_labels(labels):
            num_frs = [l.shape[0] for l in labels]
            max_t = max(num_frs)
            shape = (len(labels), max_t, labels[0].shape[1])
            out_label = np.full(shape, -100, dtype=np.int32)
            for e, l in enumerate(labels):
                out_label[e, :l.shape[0], :] = l
            return out_label, num_frs

        def gen_ctc_labels(labels):
            label_list=[]
            label_lens=[]
            for l_n in labels:
                l_l = [x[0] for x in groupby(list(l_n))]
                label_list.extend(l_l)
                label_lens.append(len(l_l))
            
            return np.array(label_list,dtype=np.int32), np.array(label_lens,dtype=np.int32)

        if self.test_only:
            utt_ids, feats = zip(*batch)
            feats, num_frs = pad_and_concat_feats(feats)
            data = {
                "utt_ids" : utt_ids,
                "num_frs": num_frs,
                "x" : torch.from_numpy(feats)
            }

        else:
            utt_ids_, feats_, labels_ = zip(*batch)
            feats, num_frs = pad_and_concat_feats(feats_)
            labels, num_labs = pad_and_concat_labels(labels_)
            assert num_labs == num_frs, "The numbers of frames and labels are not equal"
            ctc_labels, ctc_label_lens = gen_ctc_labels(labels_)
            data = {
                "utt_ids" : utt_ids_,
                "num_frs": torch.from_numpy(np.array(num_frs,dtype=np.int32)),
                "x" : torch.from_numpy(feats),
                "y" : torch.from_numpy(labels),
                "ctc_labels": torch.from_numpy(ctc_labels).view(-1),
                "ctc_label_lens": torch.from_numpy(ctc_label_lens)
            }

        return data 
