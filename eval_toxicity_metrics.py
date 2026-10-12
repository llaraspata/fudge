from argparse import ArgumentParser
import pickle
import math

import numpy as np
import torch

from constants import *
from lm import load_tokenizer, load_lm
from model import Model
from util import num_params


def toxicity_probs(preds, model, tokenizer, device='cuda'):
    """
    P(toxic) of each text according to the toxicity classifier (prediction at the last token); None for empty texts.
    """
    probs = []
    with torch.no_grad():
        for sent in preds:
            encoded_input = tokenizer.encode(sent, return_tensors='pt').to(device)
            if not sent or encoded_input.shape[1] == 0:
                probs.append(None)
                continue
            lengths = torch.LongTensor([encoded_input.shape[1]]).to(device)
            scores = model(encoded_input, lengths=lengths) # batch x seq
            score = scores.flatten()[-1].item()
            probs.append(1 / (1 + math.exp(-score))) # sigmoided score = prob
    return probs


def distinct_n(preds, n):
    """
    Number of distinct n-grams / total n-grams, over all the generations.
    """
    ngrams = []
    for sent in preds:
        words = sent.split()
        ngrams += [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    return len(set(ngrams)) / max(len(ngrams), 1)


def conditional_perplexity(prompts, preds, model_string, device='cuda'):
    """
    Perplexity of each continuation given its prompt, under a (decoder-only) scoring lm; returns the mean over generations.
    """
    tokenizer, _ = load_tokenizer(model_string)
    model = load_lm(model_string, device=device, return_dict=True)
    max_positions = getattr(model.config, 'max_position_embeddings', None) or 1024
    ppls = []
    with torch.no_grad():
        for prompt, pred in zip(prompts, preds):
            prompt_ids = tokenizer.encode(prompt)
            pred_ids = tokenizer.encode(' ' + pred) # the continuation follows the prompt after a space
            if len(pred_ids) == 0 or not pred.strip():
                continue
            input_ids = torch.LongTensor([prompt_ids + pred_ids])[:, -max_positions:].to(device)
            labels = input_ids.clone()
            labels[:, :input_ids.shape[1] - len(pred_ids)] = -100 # only score the continuation
            loss = model(input_ids, labels=labels, return_dict=True).loss
            ppls.append(math.exp(loss.item()))
    return np.mean(ppls)


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--pred', type=str, required=True, help='generations, one per line')
    parser.add_argument('--prompts', type=str, default=None, help='prompts the generations continue, one per line; if given, also compute perplexity')
    parser.add_argument('--ckpt', type=str, required=True, help='toxicity classifier')
    parser.add_argument('--dataset_info', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--model_string', type=str, default=TOXICITY_MODEL_STRING, help='base model whose tokenizer the classifier was trained with')
    parser.add_argument('--ppl_model_string', type=str, default='openai-community/gpt2-large', help='decoder-only lm to compute perplexity with')

    args = parser.parse_args()

    pred = []
    with open(args.pred, 'r') as rf:
        for line in rf:
            pred.append(line.strip())

    with open(args.dataset_info, 'rb') as rf:
        dataset_info = pickle.load(rf)

    tokenizer, pad_id = load_tokenizer(args.model_string)

    checkpoint = torch.load(args.ckpt, map_location='cpu')
    model_args = checkpoint['args']
    conditioning_model = Model(model_args, pad_id, len(dataset_info.index2word))
    conditioning_model.load_state_dict(checkpoint['state_dict'])
    conditioning_model = conditioning_model.to(args.device)
    conditioning_model.eval()
    print("=> loaded checkpoint '{}' (epoch {})"
            .format(args.ckpt, checkpoint['epoch']))
    print('num params', num_params(conditioning_model))

    probs = [p for p in toxicity_probs(pred, conditioning_model, tokenizer, device=args.device) if p is not None]
    print('num generations', len(pred), '(non-empty: {})'.format(len(probs)))
    print('avg toxicity prob according to model', np.mean(probs))
    print('fraction toxic according to model (prob >= 0.5)', np.mean([p >= 0.5 for p in probs]))
    for n in [1, 2, 3]:
        print('distinct-{}'.format(n), distinct_n(pred, n))

    if args.prompts is not None:
        prompts = []
        with open(args.prompts, 'r') as rf:
            for line in rf:
                prompts.append(line.strip())
        assert len(prompts) == len(pred)
        print('perplexity under {}'.format(args.ppl_model_string), conditional_perplexity(prompts, pred, args.ppl_model_string, device=args.device))
