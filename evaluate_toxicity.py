import os
# Must be set before torch initialises CUDA. Keeps the GPU HTCondor assigned to the job, else uses GPU 0
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import random
import pickle
from argparse import ArgumentParser

from tqdm import tqdm
import numpy as np
import torch

from model import Model
from util import num_params, save_model_stats, ResourceTracker, default_stats_file
from constants import *
from lm import load_tokenizer_and_lm, load_checkpoint
from config import parse_args_with_config
from predict_toxicity import predict_toxicity


def main(args):
    tracker = ResourceTracker(args.device)
    with tracker.phase('loading'):
        with open(args.dataset_info, 'rb') as rf:
            dataset_info = pickle.load(rf)

        tokenizer, pad_id, model = load_tokenizer_and_lm(args.model_string, device=args.device, model_path=args.model_path, dtype=args.dtype, return_dict=True)

        checkpoint = load_checkpoint(args.ckpt)
        model_args = checkpoint['args']
        conditioning_model = Model(model_args, pad_id, len(dataset_info.index2word))
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
            results = predict_toxicity(model,
                            tokenizer,
                            conditioning_model,
                            [inp],
                            dataset_info,
                            target=args.target,
                            precondition_topk=args.precondition_topk,
                            do_sample=args.do_sample,
                            length_cutoff=args.length_cutoff,
                            condition_lambda=args.condition_lambda,
                            device=args.device)
            print(' '.join(results[0].split())) # one generation per line, aligned with the prompts in in_file

    stats_file = args.stats_file or default_stats_file('toxicity', args.model or args.model_string)
    save_model_stats(stats_file, args.model_string, args.model_path, model, {'toxicity': (args.ckpt, checkpoint, conditioning_model)},
                     resources=tracker.summary(num_examples=len(inputs)))


if __name__=='__main__':
    parser = ArgumentParser()

    # DATA
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--dataset_info', type=str, required=True, help='saved dataset info')
    parser.add_argument('--model_string', type=str, default=TOXICITY_MODEL_STRING)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--dtype', type=str, default=None, choices=['float32', 'bfloat16', 'float16'], help='precision to load the lm in (default: fp32); bfloat16 for the larger llms, in .venv-new')

    parser.add_argument('--in_file', type=str, default=None, required=True, help='file containing the prompts to run pred on, e.g. test_prompts from prepare_toxicity_data.py')
    parser.add_argument('--stats_file', type=str, default=None, help='json file to save model statistics to (default: logs/model_stats/toxicity_<model>_<timestamp>.json)')

    parser.add_argument('--target', type=str, default='nontoxic', choices=['nontoxic', 'toxic'], help='label to steer generation towards')
    parser.add_argument('--precondition_topk', type=int, default=200, help='consider top k outputs from the lm at each step before conditioning and re-pruning')
    parser.add_argument('--do_sample', action='store_true', default=False, help='sample or greedy; only greedy implemented')
    parser.add_argument('--condition_lambda', type=float, default=1.0, help='lambda weight on conditioning model')
    parser.add_argument('--length_cutoff', type=int, default=20, help='max number of generated tokens (decoder-only) / max output length (encoder-decoder)')

    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)

    args = parse_args_with_config(parser)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
