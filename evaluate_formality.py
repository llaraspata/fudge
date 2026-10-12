import os
import random
import time
import pickle
import math
from argparse import ArgumentParser
from collections import namedtuple

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import Dataset
from model import Model
from util import save_checkpoint, ProgressMeter, AverageMeter, num_params, save_model_stats, ResourceTracker, default_stats_file
from constants import *
from lm import load_tokenizer, load_tokenizer_and_lm, load_checkpoint
from config import parse_args_with_config
from predict_formality import predict_formality


def main(args):
    tracker = ResourceTracker(args.device)
    with tracker.phase('loading'):
        with open(args.dataset_info, 'rb') as rf:
            dataset_info = pickle.load(rf)

        tokenizer, pad_id, model = load_tokenizer_and_lm(args.model_string, device=args.device, model_path=args.model_path, dtype=args.dtype, return_dict=True)

        checkpoint = load_checkpoint(args.ckpt)
        model_args = checkpoint['args']
        conditioning_model = Model(model_args, pad_id, len(dataset_info.index2word)) # no need to get the glove embeddings when reloading since they're saved in model ckpt anyway
        conditioning_model.load_state_dict(checkpoint['state_dict'])
        conditioning_model = conditioning_model.to(args.device)
        conditioning_model.eval()

    if args.verbose:
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.ckpt, checkpoint['epoch']))
        print('num params', num_params(conditioning_model))

    inputs = []
    with open(args.in_file, 'r') as rf:
        for line in rf:
            inputs.append(line.strip())
    
    with tracker.phase('generation'):
        for inp in tqdm(inputs, total=len(inputs)):
            results = predict_formality(model, 
                            tokenizer, 
                            conditioning_model, 
                            [inp], 
                            dataset_info, 
                            precondition_topk=args.precondition_topk,
                            do_sample=args.do_sample,
                            length_cutoff=args.length_cutoff,
                            condition_lambda=args.condition_lambda,
                            device=args.device,
                            prompt_template=args.prompt_template,
                            stop_at_newline=True)
            print(results[0])

    stats_file = args.stats_file or default_stats_file('formality', args.model or args.model_string)
    save_model_stats(stats_file, args.model_string, args.model_path, model, {'formality': (args.ckpt, checkpoint, conditioning_model)},
                     resources=tracker.summary(num_examples=len(inputs)))


if __name__=='__main__':
    parser = ArgumentParser()

    # DATA
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--dataset_info', type=str, required=True, help='saved dataset info')
    parser.add_argument('--model_string', type=str, default='Helsinki-NLP/opus-mt-es-en')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--dtype', type=str, default=None, choices=['float32', 'bfloat16', 'float16'], help='precision to load the lm in (default: fp32); bfloat16 for the larger llms, in .venv-new')

    parser.add_argument('--in_file', type=str, default=None, required=True, help='file containing text to run pred on')
    parser.add_argument('--stats_file', type=str, default=None, help='json file to save model statistics to (default: logs/model_stats/formality_<model>_<timestamp>.json)')

    parser.add_argument('--precondition_topk', type=int, default=200, help='consider top k outputs from gpt at each step before conditioning and re-pruning')
    parser.add_argument('--do_sample', action='store_true', default=False, help='sample or greedy; only greedy implemented')
    parser.add_argument('--condition_lambda', type=float, default=1.0, help='lambda weight on conditioning model')
    parser.add_argument('--length_cutoff', type=int, default=512, help='max length')
    parser.add_argument('--prompt_template', type=str, default=FORMALITY_PROMPT_TEMPLATE, help='decoder-only lms only: prompt the input is put in at {input}; generation stops at the first newline. Ignored for encoder-decoder lms (marian)')

    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)

    args = parse_args_with_config(parser)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)