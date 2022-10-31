#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Author: Bin Wang
# -----
# Copyright (c) 2022 National University of Singapore
# -----

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


from utils import label_smoothed_nll_loss


class CTRLenModel(nn.Module):

    def __init__(self, args, config):
        '''initialization'''
        super(CTRLenModel, self).__init__()

        self.seq2seq_model = AutoModelForSeq2SeqLM.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    cache_dir=args.cache_dir,
        )

        self.args            = args
        self.config          = self.seq2seq_model.config
        self.generate        = self.seq2seq_model.generate
        self.from_pretrained = self.seq2seq_model.from_pretrained
        self.save_pretrained = self.seq2seq_model.save_pretrained

        self.sim_loss        = args.sim_loss
        self.label_smoothing = args.label_smoothing


    def forward(self, batch, tokenizer):
        '''
            batch computation
        '''

        gold_len = batch.pop('gold_len') - 1
        outputs = self.seq2seq_model(**batch,output_hidden_states=True)
        
        # label smoothing loss for addtional embeddings
        if not self.label_smoothing:
            loss = outputs.loss
        else:
            output_logits = outputs.logits
            output_probs = torch.nn.functional.log_softmax(output_logits, dim=-1)
            output_probs = output_probs.view(-1, self.config.vocab_size)


            gt_logits = batch['labels']
            gt_logits = gt_logits.view(-1)

            loss, _ = label_smoothed_nll_loss(output_probs, gt_logits, self.label_smoothing, ignore_index=tokenizer.pad_token_id)

        # sim loss for addtional embeddings
        if self.sim_loss:
            one_side_window_width = int((self.args.sim_window_size - 1)/2)

            if 'bart' in self.args.model_type:
                special_token_weights = self.seq2seq_model.model.shared.weight[tokenizer.additional_special_tokens_ids]
            else:
                special_token_weights = self.seq2seq_model.shared.weight[tokenizer.additional_special_tokens_ids]
            
            special_token_weights = nn.functional.normalize(special_token_weights, dim=1)
        
            cos_sim_matrix = torch.matmul(special_token_weights,special_token_weights.T)
            
            sim_loss = 0
            for i in range(-one_side_window_width, one_side_window_width+1):
                if i == 0: continue
                sim_loss += torch.diagonal(cos_sim_matrix, offset=i).sum()

            sim_loss = cos_sim_matrix.sum() - 2 * sim_loss
            sim_loss = sim_loss / cos_sim_matrix.shape[0] **2
            loss += self.sim_loss * sim_loss

        else:
            assert False, "sim_loss for CTRLen model has to be larger than zero."

        return outputs, loss


