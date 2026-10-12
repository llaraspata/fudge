import os
import random
import pickle
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn

from model import Model
from util import num_params
from constants import *
from lm import load_tokenizer_and_lm, load_checkpoint
from config import parse_args_with_config
from predict_formality import predict_formality


def main(args):
    with open(args.dataset_info, 'rb') as rf:
        dataset_info = pickle.load(rf)
    tokenizer, pad_id, model = load_tokenizer_and_lm(args.model_string, device=args.device, dtype=args.dtype, return_dict=True)

    checkpoint = load_checkpoint(args.ckpt, map_location=args.device)
    model_args = checkpoint['args']
    conditioning_model = Model(model_args, pad_id, len(dataset_info.index2word))
    conditioning_model.load_state_dict(checkpoint['state_dict'])
    conditioning_model = conditioning_model.to(args.device)
    conditioning_model.eval()
    print("=> loaded checkpoint '{}' (epoch {})"
            .format(args.ckpt, checkpoint['epoch']))
    print('num params', num_params(conditioning_model))

    while True:
        results = predict_toxicity(model,
                        tokenizer,
                        conditioning_model,
                        [args.input_text],
                        dataset_info,
                        target=args.target,
                        precondition_topk=args.precondition_topk,
                        do_sample=args.do_sample,
                        length_cutoff=args.length_cutoff,
                        condition_lambda=args.condition_lambda,
                        device=args.device)
        print(results)
        import pdb; pdb.set_trace()


class TargetLogits(nn.Module):
    """
    Wraps the toxicity classifier (logit of toxic) so it returns the logit of the target label:
    the decoding loop turns it into log P(target | prefix), and logit(nontoxic) = -logit(toxic).
    """
    def __init__(self, conditioning_model, target):
        super(TargetLogits, self).__init__()
        assert target in ['toxic', 'nontoxic']
        self.conditioning_model = conditioning_model
        self.sign = 1 if target == 'toxic' else -1

    def forward(self, *args, **kwargs):
        return self.sign * self.conditioning_model(*args, **kwargs)


def predict_toxicity(model, tokenizer, conditioning_model, input_text, dataset_info, target='nontoxic', precondition_topk=200, do_sample=False, length_cutoff=20, condition_lambda=1.0, device='cuda'):
    """
    Generate with FUDGE steering towards target ('nontoxic' or 'toxic').
    Works with decoder-only lms (e.g. gpt2: input_text is the prompt, returns its continuation)
    and encoder-decoder ones (e.g. marian: input_text is the source, returns the generated output);
    the classifier only sees the generated tokens, as it was trained on standalone texts.
    """
    return predict_formality(model,
                             tokenizer,
                             TargetLogits(conditioning_model, target),
                             input_text,
                             dataset_info,
                             precondition_topk=precondition_topk,
                             do_sample=do_sample,
                             length_cutoff=length_cutoff,
                             condition_lambda=condition_lambda,
                             device=device)


if __name__=='__main__':
    parser = ArgumentParser()

    # DATA
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--dataset_info', type=str, required=True, help='saved dataset info')
    parser.add_argument('--model_string', type=str, default=TOXICITY_MODEL_STRING)
    parser.add_argument('--dtype', type=str, default=None, choices=['float32', 'bfloat16', 'float16'], help='precision to load the lm in (default: fp32); bfloat16 for the larger llms, in .venv-new')

    parser.add_argument('--input_text', type=str, default=None, required=True, help='text to run pred on')

    parser.add_argument('--target', type=str, default='nontoxic', choices=['nontoxic', 'toxic'], help='label to steer generation towards')
    parser.add_argument('--precondition_topk', type=int, default=200, help='consider top k outputs from the lm at each step before conditioning and re-pruning')
    parser.add_argument('--do_sample', action='store_true', default=False, help='sample instead of greedy')
    parser.add_argument('--condition_lambda', type=float, default=1.0, help='lambda weight on conditioning model')
    parser.add_argument('--length_cutoff', type=int, default=20, help='max number of generated tokens (decoder-only) / max output length (encoder-decoder)')

    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--debug', action='store_true', default=False)

    args = parse_args_with_config(parser)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
