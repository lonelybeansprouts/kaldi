# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import List, Optional, Tuple

import torch

from torch.nn.utils.rnn import pad_sequence

from pynet.transformer.cmvn import GlobalCMVN
from pynet.transformer.ctc import CTC
from pynet.transformer.decoder import (TransformerDecoder,
                                       BiTransformerDecoder)
from pynet.transformer.encoder import ConformerEncoder
from pynet.transformer.encoder import TransformerEncoder
from pynet.transformer.label_smoothing_loss import LabelSmoothingLoss
from pynet.utils.cmvn import load_cmvn
from pynet.utils.common import (IGNORE_ID, add_sos_eos, log_add,
                                remove_duplicates_and_blank, th_accuracy,
                                reverse_pad_list)
from pynet.utils.mask import (make_pad_mask, mask_finished_preds,
                              mask_finished_scores, subsequent_mask)

from . import e2e_pdf_ce_multiattention
from . import e2e_pdf_bfctc_multiattention
from . import e2e_pdf_random_bfctc_multiattention


def init_asr_model(configs):
    task_type = configs['task_type']
    if task_type == 'e2e_pdf_ce_multiattention':
        model = e2e_pdf_ce_multiattention.init_asr_model(configs)
    elif task_type == 'e2e_pdf_bfctc_multiattention':
        model = e2e_pdf_bfctc_multiattention.init_asr_model(configs)
    elif task_type == 'e2e_pdf_random_bfctc_multiattention':
        model = e2e_pdf_random_bfctc_multiattention.init_asr_model(configs)
    else:
        assert False, "not support task_type: " + task_type
    return model
