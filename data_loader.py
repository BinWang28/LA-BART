#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Author: Bin Wang
# -----
# Copyright (c) 2022 National University of Singapore
# -----

import json
import csv
import random

import datasets
from datasets import Dataset
from torch.utils.data import DataLoader

from transformers import DataCollatorForSeq2Seq

import utils



def raw_data_loader(args):
    ''' load raw datasets from csv files '''

    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    if args.test_file is not None:
        data_files["test"] = args.test_file

    if 'samsum' in args.train_file:
        train_dict = load_from_samsum(args, args.train_file)
        val_dict   = load_from_samsum(args, args.validation_file)
        test_dict  = load_from_samsum(args, args.test_file)

    elif 'dialogsum' in args.train_file:
        train_dict = load_from_dialogsum(args, args.train_file)
        val_dict   = load_from_dialogsum(args, args.validation_file)
        test_dict  = load_from_dialogsum(args, args.test_file)

    train_dict = utils.len_adjust(args, train_dict, 'train')
    val_dict   = utils.len_adjust(args, val_dict, 'val')
    test_dict  = utils.len_adjust(args, test_dict, 'test')

    raw_datasets = datasets.DatasetDict({"train":train_dict, "validation":val_dict, "test":test_dict})

    return raw_datasets


def load_from_samsum(args, file_path):
    ''' load samsum csv data '''

    id_list       = []
    dialogue_list = []
    summary_list  = []

    with open(file_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            id_list.append(row['id'])
            dialogue_list.append(row['dialogue'])
            summary_list.append(row['summary'])

    data_dict = {
                'id'      : id_list,
                'dialogue': dialogue_list,
                'summary' : summary_list
                 }

    data_dict = Dataset.from_dict(data_dict)

    return data_dict


def load_from_dialogsum(args, file_path):
    ''' load dialogue jsonl data '''

    data = []
    
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    id_list       = [sample['fname'] for sample in data]
    dialogue_list = [sample['dialogue'] for sample in data]

    if 'summary' in data[0]:
        summary_list  = [sample['summary'] for sample in data]

    elif 'summary1' in data[0]:

        id_list1 = [id+"_sum1" for id in id_list]
        id_list2 = [id+"_sum2" for id in id_list]
        id_list3 = [id+"_sum3" for id in id_list]

        id_list = id_list1 + id_list2 + id_list3
        dialogue_list = dialogue_list + dialogue_list + dialogue_list

        summary_list1  = [sample['summary1'] for sample in data]
        summary_list2  = [sample['summary2'] for sample in data]
        summary_list3  = [sample['summary3'] for sample in data]

        summary_list = summary_list1 + summary_list2 + summary_list3

    data_dict = {'id': id_list,
                'dialogue': dialogue_list,
                'summary': summary_list}

    data_dict = Dataset.from_dict(data_dict)

    return data_dict


def data_processor(logger, args, accelerator, raw_datasets, tokenizer, model):
    ''' prepare dataset format for train/val/test '''
    def preprocess_function(examples):

        # summary - target
        targets = examples[summary_column]
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        if args.ctrlen_model:
            gold_sum_len = [len(item) for item in labels['attention_mask']]

        # dialogue - input
        inputs = examples[text_column]
        new_inputs = []
        for i, inp in enumerate(inputs):
            if args.ctrlen_model:
                if 'pred_len' in examples:
                    new_inputs.append(prefix + "<len_{}> ".format(examples['pred_len'][i]) + inp)

                else:
                    new_inputs.append(prefix + "<len_{}> ".format(gold_sum_len[i]) + inp)
            else:
                new_inputs.append(prefix + inp)

        inputs = new_inputs
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        
        model_inputs["labels"] = labels["input_ids"]

        if args.ctrlen_model:
            model_inputs["gold_len"] = gold_sum_len

        return model_inputs

    prefix = args.source_prefix if args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names

    # Get the column names for input/target.
    text_column = args.text_column
    if text_column not in column_names:
        raise ValueError(
            f"--text_column' value '{args.text_column}' needs to be one of: {', '.join(column_names)}"
        )

    summary_column = args.summary_column
    if summary_column not in column_names:
        raise ValueError(
            f"--summary_column' value '{args.summary_column}' needs to be one of: {', '.join(column_names)}"
        )

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            batch_size=1000,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset  = processed_datasets["validation"]
    test_dataset  = processed_datasets["test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_test_batch_size)

    return (train_dataloader, eval_dataloader, test_dataloader), (train_dataset, eval_dataset, test_dataset)



