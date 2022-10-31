#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Author: Bin Wang
# -----
# Copyright (c) 2022 National University of Singapore
# -----

import os 
import numpy as np
import torch
from model import CTRLenModel

from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

def model_loader(accelerator, logger, args):
    '''
        load transformer models (config, tokenizer, s2s model)
    '''

    # model config
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    # tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    
    ori_tokenizer_len = len(tokenizer)

    # model
    if args.ctrlen_model:

        special_tokens_dict = {'additional_special_tokens': ['<len_{}>'.format(i) for i in range(args.len_start, args.len_end+1)]}
        tokenizer.add_special_tokens(special_tokens_dict)

        model = CTRLenModel(args, config)
        model.seq2seq_model.resize_token_embeddings(ori_tokenizer_len)
        model.seq2seq_model.resize_token_embeddings(len(tokenizer))

        if args.special_len_token_init == 'random':
            pass
        
        elif args.special_len_token_init == 'zero':

            if 'bart' in args.model_type: 
                emavg_embs = torch.zeros(len(special_tokens_dict['additional_special_tokens']), model.seq2seq_model.model.shared.weight.shape[1], dtype=torch.float) 
                with torch.no_grad(): model.seq2seq_model.model.shared.weight[tokenizer.additional_special_tokens_ids] = emavg_embs
            else:
                emavg_embs = torch.zeros(len(special_tokens_dict['additional_special_tokens']), model.seq2seq_model.shared.weight.shape[1], dtype=torch.float) 
                with torch.no_grad(): model.seq2seq_model.shared.weight[tokenizer.additional_special_tokens_ids] = emavg_embs

        
        elif args.special_len_token_init == 'token_embs':
            token2ids = []
            for i in range(args.len_start, args.len_end+1):
                token2id = tokenizer.convert_tokens_to_ids(str(i+1))
                token2ids.append(token2id)

            if 'bart' in args.model_type:
                emavg_embs = model.seq2seq_model.model.shared.weight[token2ids].cpu().detach()
                with torch.no_grad():
                    model.seq2seq_model.model.shared.weight[tokenizer.additional_special_tokens_ids] = emavg_embs
            else:
                emavg_embs = model.seq2seq_model.shared.weight[token2ids].cpu().detach()
                with torch.no_grad():
                    model.seq2seq_model.shared.weight[tokenizer.additional_special_tokens_ids] = emavg_embs
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )

        model.resize_token_embeddings(ori_tokenizer_len)
        model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")


    # Save initial model
    if args.output_dir is not None:
        os.makedirs(args.output_dir+'/start', exist_ok=True)
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir+'/start', save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir+'/start')


        vocab = tokenizer.vocab.copy()
        vocab = {k: v for k, v in sorted(vocab.items(), key=lambda item: item[1])}
        with open(args.output_dir + '/start/vocab.txt', 'w') as f:
            for word, index in vocab.items():
                word = word.encode('ascii', 'ignore').decode('ascii')
                f.write(str(index) + ': ' + word + '\n')

    # save cosine similarity for CTRLenModel model
    if args.ctrlen_model:
        from sklearn.metrics.pairwise import cosine_similarity

        if 'bart' in args.model_type: 
            extra_embedding = model.seq2seq_model.model.shared.weight[tokenizer.additional_special_tokens_ids].cpu().detach().numpy()#[-add_len_num:]
        else:
            extra_embedding = model.seq2seq_model.shared.weight[tokenizer.additional_special_tokens_ids].cpu().detach().numpy()#[-add_len_num:]

        pair_wise_cosine = cosine_similarity(extra_embedding)
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(15,15))
        cax = ax.matshow(pair_wise_cosine)

        ax.grid(True)
        labels = [*range(args.len_start,pair_wise_cosine.shape[0]+args.len_start)]
        plt.title('length token similarity')
        plt.xticks(range(len(labels)), labels, rotation=90);
        plt.yticks(range(len(labels)), labels);
        fig.colorbar(cax, ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, .75,.8,.85,.90,.95,1])

        fig.dpi = 300
        fig.savefig(args.output_dir+'/start/diaglen_sumlen.png', dpi=fig.dpi)

    return config, tokenizer, model


def numpy_ewma_vectorized(alpha, data):
    '''
        exponential moving average for 1d vector
    '''

    alpha_rev = 1-alpha
    n = data.shape[0]

    pows = alpha_rev**(np.arange(n+1))

    scale_arr = 1 / pows[:-1]


    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)

    
    #mult = data*(pw0*scale_arr)[:,np.newaxis]

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum(axis=0)
    
    out = offset + cumsums*scale_arr[::-1]
    return out

def numpy_window_avg(data, window_size, alpha):
    '''
        weighted average within a window
    '''
    num_samples = data.shape[0]
    half_win_length = int((window_size-1)/2)

    new_data = []
    window_weigting = alpha**abs(np.arange(-half_win_length, half_win_length+1))
    window_weigting = window_weigting / np.sum(window_weigting)
    
    for index in range(num_samples):
        if index < half_win_length:
            partial_data = data[0:index+half_win_length+1,:]
            add_data = partial_data[::-1][:window_size-partial_data.shape[0]]
            full_data = np.concatenate((add_data,partial_data),axis=0)
        elif index >= (num_samples - half_win_length):
            partial_data = data[index-half_win_length:]
            add_data = partial_data[::-1][partial_data.shape[0]-window_size:]
            full_data = np.concatenate((partial_data,add_data),axis=0)
        else:
            full_data = data[index-half_win_length:index+half_win_length+1]

        new_data.append(window_weigting.dot(full_data))
    
    new_data = np.stack(new_data)

    return new_data



