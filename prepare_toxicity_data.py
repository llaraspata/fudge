"""
Turn RealToxicityPrompts (prompts.jsonl, https://huggingface.co/datasets/allenai/real-toxicity-prompts) into the
layout data.py reads for the toxicity task:

    out_dir/{train,val,test}/{toxic,nontoxic}   one text per line
    out_dir/test_prompts                        the test prompt texts, one per line, to generate continuations from

Each record has a prompt and a continuation, each with its own perspective api toxicity score; each of them becomes
one example, labeled toxic if its score >= TOXICITY_THRESHOLD (texts without a score are skipped).
Records are shuffled and split 70/10/20 by record, so a prompt and its continuation always land in the same split.
"""
import os
import json
import random
from argparse import ArgumentParser

from constants import *


def clean(text):
    return ' '.join(text.split()) # one example per line, so no newlines inside a text


def main(args):
    records = []
    with open(args.in_file, 'r') as rf:
        for line in rf:
            if line.strip():
                records.append(json.loads(line))

    random.seed(args.seed)
    random.shuffle(records)
    num_train = int(len(records) * args.train_frac)
    num_val = int(len(records) * args.val_frac)
    splits = {
        'train': records[:num_train],
        'val': records[num_train:num_train + num_val],
        'test': records[num_train + num_val:],
    }

    for split, split_records in splits.items():
        texts = {'toxic': [], 'nontoxic': []}
        skipped = 0
        for record in split_records:
            for field in args.fields:
                part = record.get(field) or {}
                text, score = clean(part.get('text') or ''), part.get('toxicity')
                if not text or score is None:
                    skipped += 1
                    continue
                texts['toxic' if score >= args.threshold else 'nontoxic'].append(text)

        os.makedirs(os.path.join(args.out_dir, split), exist_ok=True)
        for category, category_texts in texts.items():
            with open(os.path.join(args.out_dir, split, category), 'w') as wf:
                for text in category_texts:
                    wf.write(text + '\n')
        print(split, 'records', len(split_records), 'toxic', len(texts['toxic']), 'nontoxic', len(texts['nontoxic']), 'skipped', skipped)

    with open(os.path.join(args.out_dir, 'test_prompts'), 'w') as wf:
        for record in splits['test']:
            text = clean((record.get('prompt') or {}).get('text') or '')
            if text:
                wf.write(text + '\n')


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--in_file', type=str, required=True, help='RealToxicityPrompts prompts.jsonl')
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--fields', type=str, nargs='+', default=['prompt', 'continuation'], choices=['prompt', 'continuation'], help='which texts of each record to use as examples')
    parser.add_argument('--threshold', type=float, default=TOXICITY_THRESHOLD, help='toxicity score at or above which a text is labeled toxic')
    parser.add_argument('--train_frac', type=float, default=0.7)
    parser.add_argument('--val_frac', type=float, default=0.1, help='the rest after train and val is test')
    parser.add_argument('--seed', type=int, default=1, help='random seed for the shuffle')
    args = parser.parse_args()

    main(args)
