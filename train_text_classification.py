#!/usr/bin/env python3
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2024 Sugar. All Rights Reserved
#
########################################################################

"""
    File: train_text_classification.py
    Desc: 文本分类训练代码
    Author: sugar(@google.com)
    Date: 2024-04-1 11:25
    desc: 纪念一下leslie 張發宗
    code refer: https://github.com/zyds/transformers-code/blob/master/01-Getting%20Started/04-model/classification_demo.ipynb
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')
device = '3'  # 本次实验需要用到的卡
os.environ["CUDA_VISIBLE_DEVICES"] = device
os.environ['CUDA_LAUNCH_BLOCKING'] = device
import torch
import torch.nn as nn
import transformers
import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification,TrainingArguments,Trainer
from datasets import load_dataset
from torch.utils.data import DataLoader,Dataset,random_split
import pandas as pd
from datasets import Dataset as HFDataset
from data_utils import TextClassification
from text_trainer import MyTrainer
from text_callback import ProgressCallback



def run(data_path, pretrained_model_name_or_path,classes):
    def process_func(examples):
        review = examples['review']
        label = examples['labels']
        inputs = tokenizer(review,max_length=128,padding="max_length",truncation=True,return_tensors='pt')
        inputs['labels'] = torch.tensor(label)
        return inputs

    # ===============================  加载数据集  =============================== #
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    dataset = TextClassification(data_path)
    # 注意dataset返回的是字典形式 方便构建huggingface dataset
    train_dataset, valid_dataset = random_split(dataset, lengths=[0.9, 0.1])
    # 转为huggingface dataset
    train_ds = HFDataset.from_list(train_dataset)
    valid_ds = HFDataset.from_list(valid_dataset)

    tokenizer_train_ds = train_ds.map(process_func,batched=True,remove_columns=train_ds.column_names)
    tokenizer_valid_ds = valid_ds.map(process_func,batched=True,remove_columns=valid_ds.column_names)

    # 加载模型 注意classes参数 这里是2分类
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path,num_labels=classes)

    # ===============================  构建Trainer  =============================== #
    args = TrainingArguments(
        output_dir="./text_classification",
        per_device_train_batch_size = 4,
        per_device_eval_batch_size = 4,
        gradient_accumulation_steps = 8,
        logging_steps = 50,
        num_train_epochs = 2,
        evaluation_strategy = 'steps',
        # 是否打开进度条
        # disable_tqdm = True,
        report_to = 'tensorboard',
        )
    
    trainer = MyTrainer(
        model=model,
        args=args,
        train_dataset=tokenizer_train_ds,
        eval_dataset=tokenizer_valid_ds,)
    # ===============================  构建CallBack  =============================== #
    progress_callback = ProgressCallback().setup(total_epochs=args.num_train_epochs,print_every=200)
    trainer.add_callback(progress_callback)

    # ===============================  训练模型  =============================== #
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

if __name__ == '__main__':
    data_path                       = 'data/ChnSentiCorp_htl_all.csv'
    pretrained_model_name_or_path   = 'dienstag/chinese-bert-wwm'
    classes                         = 2
    run(data_path = data_path, pretrained_model_name_or_path = pretrained_model_name_or_path, classes = classes)
