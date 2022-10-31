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
from pynet.transformer.label_bfctc_adversary_loss import LabelBfCtcAdversaryLoss
from pynet.utils.cmvn import load_cmvn
from pynet.utils.common import (IGNORE_ID, add_sos_eos, log_add,
                                remove_duplicates_and_blank, th_accuracy,
                                reverse_pad_list)
from pynet.utils.mask import (make_pad_mask, mask_finished_preds,
                              mask_finished_scores, subsequent_mask)


class ASRModel(torch.nn.Module):
    """CTC-attention hybrid Encoder-Decoder model"""
    def __init__(
        self,
        encoder: TransformerEncoder,
        pdf_decoder: TransformerDecoder,
        phone_decoder: TransformerDecoder,
        vocab_decoder: TransformerDecoder,
        pdf_size: int,
        phone_size: int,
        vocab_size: int,
        pdf_attention_weight: float,
        phone_attention_weight: float,
        vacab_attention_weight: float,
        pdf_bfctc_weight: float,
        adversary_weight: float = 0.1,
        ignore_id: int = IGNORE_ID,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
    ):

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        # self.sos = vocab_size - 1
        # self.eos = vocab_size - 1
        self.pdf_size = pdf_size
        self.phone_size = phone_size
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id

        self.encoder = encoder
        self.pdf_decoder = pdf_decoder
        self.phone_decoder = phone_decoder
        self.vocab_decoder = vocab_decoder

        self.pdf_bfctc_weight = pdf_bfctc_weight
        self.pdf_attention_weight = pdf_attention_weight
        self.phone_attention_weight = phone_attention_weight
        self.vacab_attention_weight = vacab_attention_weight


        self.pdf_bfctc_loss = LabelBfCtcAdversaryLoss(
            size=pdf_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            adversary_weight=adversary_weight,
            normalize_length=length_normalized_loss,
            input_size = self.encoder.output_size(),
        )
        self.pdf_attention_loss = LabelSmoothingLoss(
            size=pdf_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )
        self.phone_attention_loss = LabelSmoothingLoss(
            size=phone_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )
        self.vocab_attention_loss = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )
    
    def unpack_train_data(self, batch_data, device):
        return batch_data['feats'].size(0), \
            (batch_data['feats'].to(device), batch_data['feats_lengths'].to(device), \
            batch_data['pdf_ali_repeat_p1'].to(device), batch_data['pdf_ali_repeat_lengths_p1'].to(device), \
            batch_data['pdf_ali_repeat_p2'].to(device), batch_data['pdf_ali_repeat_lengths_p2'].to(device), \
            batch_data['pdf_ali_none_repeat'].to(device), batch_data['pdf_ali_none_repeat_lengths'].to(device), \
            batch_data['phone_ali_none_repeat'].to(device), batch_data['phone_ali_none_repeat_lengths'].to(device), \
            batch_data['labels'].to(device), batch_data['label_lengths'].to(device))

    def unpack_test_data(self, batch_data, device):
        return batch_data['feats'].size(0), (batch_data['feats'].to(device), batch_data['feats_lengths'].to(device))

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        pdf_ali_repeat_p1: torch.Tensor,
        pdf_ali_repeat_lengths_p1: torch.Tensor,
        pdf_ali_repeat_p2: torch.Tensor,
        pdf_ali_repeat_lengths_p2: torch.Tensor,
        pdf_ali_no_repeat: torch.Tensor,
        pdf_ali_no_repeat_lengths: torch.Tensor,
        phone_ali_no_repeat: torch.Tensor,
        phone_ali_no_repeat_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor
    ):
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            ...
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (speech.shape[0] == speech_lengths.shape[0] == 
                text.shape[0] == text_lengths.shape[0]==
                pdf_ali_repeat_p1.shape[0] == pdf_ali_repeat_lengths_p1.shape[0] ==
                pdf_ali_repeat_p2.shape[0] == pdf_ali_repeat_lengths_p2.shape[0] ==
                pdf_ali_no_repeat.shape[0]==pdf_ali_no_repeat_lengths.shape[0]==
                phone_ali_no_repeat.shape[0]==phone_ali_no_repeat_lengths.shape[0]), (speech.shape, speech_lengths.shape,
                text.shape, text_lengths.shape, pdf_ali_no_repeat.shape, pdf_ali_no_repeat_lengths.shape,
                phone_ali_no_repeat.shape, phone_ali_no_repeat_lengths.shape)
        
        assert (speech.dtype == torch.float32)
        assert (speech_lengths.dtype == torch.int32)
        assert (pdf_ali_repeat_p1.dtype == torch.int32)
        assert (pdf_ali_repeat_lengths_p1.dtype == torch.int32)
        assert (pdf_ali_repeat_p2.dtype == torch.int32)
        assert (pdf_ali_repeat_lengths_p2.dtype == torch.int32)
        assert (pdf_ali_no_repeat.dtype == torch.int32)
        assert (pdf_ali_no_repeat_lengths.dtype == torch.int32)
        assert (phone_ali_no_repeat.dtype == torch.int32)
        assert (phone_ali_no_repeat_lengths.dtype == torch.int32)
        assert (text.dtype == torch.int32)
        assert (text_lengths.dtype == torch.int32)

        # 1. Encoder
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        # encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        pdf_bfctc_loss_, pdf_bfctc_loss_p1_, pdf_bfctc_loss_p2_ = self.pdf_bfctc_loss(encoder_out, pdf_ali_repeat_p1, pdf_ali_repeat_p2, speech_lengths, pdf_ali_repeat_lengths_p1, pdf_ali_repeat_lengths_p2) 

        if self.pdf_attention_weight > 0.0:
            pdf_attention_loss_= self._calc_att_loss(self.pdf_decoder, self.pdf_attention_loss, encoder_out, encoder_mask, pdf_ali_no_repeat, pdf_ali_no_repeat_lengths, self.pdf_size-1, self.pdf_size-1)
        else:
            pdf_attention_loss_ = torch.FloatTensor([0.0]).to(speech)
        
        if self.phone_attention_weight > 0.0:
            phone_attention_loss_ = self._calc_att_loss(self.phone_decoder, self.phone_attention_loss, encoder_out, encoder_mask, phone_ali_no_repeat, phone_ali_no_repeat_lengths, self.phone_size-1, self.phone_size-1)
        else:
            phone_attention_loss_ = torch.FloatTensor([0.0]).to(speech)
        
        if self.vacab_attention_weight > 0.0:
            vacab_attention_loss_= self._calc_att_loss(self.vocab_decoder, self.vocab_attention_loss, encoder_out, encoder_mask, text, text_lengths, self.vocab_size-1, self.vocab_size-1)
        else:
            vacab_attention_loss_ = torch.FloatTensor([0.0]).to(speech)

        loss = self.pdf_bfctc_weight * (pdf_bfctc_loss_)  +  self.pdf_attention_weight*pdf_attention_loss_ + self.phone_attention_weight*phone_attention_loss_ + self.vacab_attention_weight*vacab_attention_loss_
        
        return loss, dict(pdf_bfctc_loss=pdf_bfctc_loss_, pdf_bfctc_loss_p1=pdf_bfctc_loss_p1_, pdf_bfctc_loss_p2=pdf_bfctc_loss_p2_, pdf_attention_loss=pdf_attention_loss_, phone_attention_loss=phone_attention_loss_, vacab_attention_loss=vacab_attention_loss_)

    def _calc_att_loss(
        self,
        decoder_module: TransformerEncoder,
        criterion_att: LabelSmoothingLoss,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        sos: int,
        eos: int
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, sos, eos,
                                            self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # reverse the seq, used for right to left decoder
        r_ys_pad = reverse_pad_list(ys_pad, ys_pad_lens, float(self.ignore_id))
        r_ys_in_pad, r_ys_out_pad = add_sos_eos(r_ys_pad, sos, eos,
                                                self.ignore_id)
        reverse_weight = 0.0 
        # 1. Forward decoder
        decoder_out, r_decoder_out, _ = decoder_module(encoder_out, encoder_mask,
                                                     ys_in_pad, ys_in_lens,
                                                     r_ys_in_pad,
                                                     reverse_weight)
        # 2. Compute attention loss
        loss_att = criterion_att(decoder_out, ys_out_pad)

        return loss_att

    def _forward_encoder(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Let's assume B = batch_size
        # 1. Encoder
        if simulate_streaming and decoding_chunk_size > 0:
            encoder_out, encoder_mask = self.encoder.forward_chunk_by_chunk(
                speech,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )  # (B, maxlen, encoder_dim)
        else:
            encoder_out, encoder_mask = self.encoder(
                speech,
                speech_lengths,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )  # (B, maxlen, encoder_dim)
        return encoder_out, encoder_mask
    
    def get_logprobs(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,        
    ):
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)
        
        return self.pdf_bfctc_loss.get_logprobs(encoder_out)


    def get_probs(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,        
    ):
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)
        
        return self.pdf_bfctc_loss.get_probs(encoder_out)


def init_asr_model(configs):
    if configs['cmvn_file'] is not None:
        mean, istd = load_cmvn(configs['cmvn_file'], configs['is_json_cmvn'])
        global_cmvn = GlobalCMVN(
            torch.from_numpy(mean).float(),
            torch.from_numpy(istd).float())
    else:
        global_cmvn = None

    input_dim = configs['input_dim']
    pdf_size = configs['pdf_size']
    phone_size = configs['phone_size'] #todo: set outsise
    vocab_size = configs['vocab_size'] #todo: set outsise

    encoder_type = configs.get('encoder', 'conformer')
    decoder_type = configs.get('decoder', 'bitransformer')

    if encoder_type == 'conformer':
        encoder = ConformerEncoder(input_dim,
                                   global_cmvn=global_cmvn,
                                   **configs['encoder_conf'])
    else:
        encoder = TransformerEncoder(input_dim,
                                     global_cmvn=global_cmvn,
                                     **configs['encoder_conf'])
    
    assert decoder_type == 'transformer'
    pdf_decoder = TransformerDecoder(pdf_size, encoder.output_size(),
                            **configs['decoder_conf'])
    phone_decoder = TransformerDecoder(phone_size, encoder.output_size(),
                            **configs['decoder_conf'])
    vocab_decoder = TransformerDecoder(vocab_size, encoder.output_size(),
                                    **configs['decoder_conf'])

    model = ASRModel(
        pdf_size=pdf_size,
        phone_size=phone_size,
        vocab_size=vocab_size,
        encoder=encoder,
        pdf_decoder=pdf_decoder,
        phone_decoder=phone_decoder,
        vocab_decoder=vocab_decoder,
        **configs['model_conf']
    )
    return model
