import os
import random
import time
import pickle
import math
from argparse import ArgumentParser

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import Dataset
from model import Model
from util import save_checkpoint, ProgressMeter, AverageMeter, num_params
from constants import *
from lm import load_tokenizer, load_tokenizer_and_lm, load_checkpoint
from config import parse_args_with_config

def main(args):
    with open(args.dataset_info, 'rb') as rf:
        dataset_info = pickle.load(rf)
    tokenizer, pad_id, model = load_tokenizer_and_lm(args.model_string, device=args.device, dtype=args.dtype, return_dict=True)

    checkpoint = load_checkpoint(args.ckpt, map_location=args.device)
    model_args = checkpoint['args']
    conditioning_model = Model(model_args, pad_id, len(dataset_info.index2word)) # no need to get the glove embeddings when reloading since they're saved in model ckpt anyway
    conditioning_model.load_state_dict(checkpoint['state_dict'])
    conditioning_model = conditioning_model.to(args.device)
    conditioning_model.eval()
    print("=> loaded checkpoint '{}' (epoch {})"
            .format(args.ckpt, checkpoint['epoch']))
    print('num params', num_params(conditioning_model))

    while True:
        results = predict_formality(model, 
                        tokenizer, 
                        conditioning_model, 
                        [args.input_text], 
                        dataset_info, 
                        precondition_topk=args.precondition_topk,
                        do_sample=args.do_sample,
                        length_cutoff=args.length_cutoff,
                        condition_lambda=args.condition_lambda,
                        device=args.device,
                        prompt_template=args.prompt_template,
                        stop_at_newline=True)
        print(results)
        import pdb; pdb.set_trace()


def predict_formality(model, tokenizer, conditioning_model, input_text, dataset_info, precondition_topk=200, do_sample=False, length_cutoff=512, condition_lambda=1.0, device='cuda', prompt_template=None, stop_at_newline=False):
    """
    Generate with FUDGE, for encoder-decoder lms (e.g. marian: input_text is the source, returns the generated output)
    and decoder-only ones (e.g. gpt2, llama, qwen, gemma: input_text is the prompt, returns its continuation).
    The conditioning model only sees the generated tokens.
    prompt_template: decoder-only lms only, string the input is put in at {input}, e.g. a few-shot translation prompt.
    stop_at_newline: decoder-only lms only, stop at the first generated newline and drop it (e.g. after one translated line).
    """
    with torch.no_grad():
        batch_size = len(input_text)
        if prompt_template is not None and not model.config.is_encoder_decoder:
            input_text = [prompt_template.replace('{input}', it) for it in input_text]

        # assumes initially all same length.
        encoded_input = [tokenizer.encode(it, return_tensors='pt').to(device) for it in input_text] # batch x seq
        encoded_input = torch.cat(encoded_input, dim=0)

        min_length = 0
        temperature = 1.0
        top_k = 50
        top_p = 1.0
        repetition_penalty = 1.0
        no_repeat_ngram_size = 0
        effective_batch_size = batch_size
        use_cache = True

        if model.config.is_encoder_decoder: # e.g. marian: input is encoded once, decoder generates from the start token
            start_token_id = model.config.decoder_start_token_id # 58100 for marian
            input_ids = torch.LongTensor([[start_token_id]] * batch_size).to(device)
            gen_start = 1 # generated text starts after the start token
            max_length = length_cutoff
            bad_words_ids = [[start_token_id]]
            pad_token_id = start_token_id
            eos_token_id = model.config.eos_token_id # 0 for marian
            attention_mask = encoded_input.new_ones(encoded_input.shape)
            model_specific_kwargs = {'encoder_outputs': model.get_encoder()(encoded_input, attention_mask=attention_mask)}
            stop_at_newline = False
        else: # decoder-only, e.g. gpt2: input is the prompt, generation continues it
            input_ids = encoded_input
            gen_start = input_ids.shape[1] # generated text starts after the prompt
            max_length = gen_start + length_cutoff
            max_positions = getattr(model.config, 'max_position_embeddings', None)
            if max_positions is not None:
                max_length = min(max_length, max_positions)
            bad_words_ids = None # PAD_TOKEN was added to the tokenizer only, the lm can't produce it
            eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else model.config.eos_token_id
            if isinstance(eos_token_id, list): # some recent configs list several
                eos_token_id = eos_token_id[0]
            pad_token_id = getattr(model.config, 'pad_token_id', None)
            if pad_token_id is None:
                pad_token_id = eos_token_id
            attention_mask = input_ids.new_ones(input_ids.shape)
            model_specific_kwargs = {}
        cur_len = input_ids.shape[1]
        predictor_vocab_size = tokenizer.convert_tokens_to_ids(PAD_TOKEN) # lm tokens from here on (e.g. qwen's unused padding rows) are unknown to the predictor
        is_stop_token = (lambda token_id: '\n' in tokenizer.decode([token_id])) if stop_at_newline else None

        output = _generate_no_beam_search(model,
                                        conditioning_model,
                                        condition_lambda,
                                        precondition_topk,
                                        input_ids,
                                        cur_len,
                                        max_length,
                                        min_length,
                                        do_sample,
                                        temperature,
                                        top_k,
                                        top_p,
                                        repetition_penalty,
                                        no_repeat_ngram_size,
                                        bad_words_ids,
                                        pad_token_id,
                                        eos_token_id,
                                        batch_size,
                                        attention_mask,
                                        use_cache,
                                        gen_start,
                                        model_specific_kwargs,
                                        predictor_vocab_size=predictor_vocab_size,
                                        is_stop_token=is_stop_token)

        if model.config.is_encoder_decoder:
            output = [s[gen_start:] for s in output] # to delete the pad token
            output = [s[:(s == eos_token_id).nonzero()[0, 0]] if (s == eos_token_id).any() else s for s in output] # recent transformers decode the eos too
            return [tokenizer.decode(s) for s in output]
        outputs = [tokenizer.decode(s[gen_start:], skip_special_tokens=True) for s in output] # only the continuation, without eos/padding
        if stop_at_newline:
            outputs = [o.split('\n')[0].strip() for o in outputs]
        return outputs


def _lm_next_token_scores(model, input_ids, past, attention_mask, model_kwargs, cur_len, min_length, max_length, no_repeat_ngram_size, bad_words_ids, eos_token_id, repetition_penalty, batch_size, use_cache):
    """
    One lm step: returns (next token scores, past).
    transformers 3.x (.venv): through the model's generation helpers, as originally.
    transformers >= 4 (.venv-new), where those helpers are gone: a plain forward with the kv cache. Only the options
    predict_formality uses are supported there (no repetition penalty / ngram blocking / min length; bad words of 1 token).
    """
    if hasattr(model, 'postprocess_next_token_scores'): # transformers 3.x
        model_inputs = model.prepare_inputs_for_generation(
            input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_kwargs
        )

        outputs = model(**model_inputs, return_dict=True)
        next_token_logits = outputs.logits[:, -1, :]

        scores = model.postprocess_next_token_scores(
            scores=next_token_logits,
            input_ids=input_ids,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            cur_len=cur_len,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            batch_size=batch_size,
            num_beams=1,
        )

        # if model has past, then set the past variable to speed up decoding
        if "past_key_values" in outputs:
            past = outputs.past_key_values
        elif "mems" in outputs:
            past = outputs.mems
        return scores, past

    assert repetition_penalty == 1.0 and no_repeat_ngram_size == 0 and min_length == 0
    new_ids = input_ids if past is None else input_ids[:, -1:] # with the cache, only feed the last token
    if model.config.is_encoder_decoder:
        outputs = model(encoder_outputs=model_kwargs['encoder_outputs'], attention_mask=attention_mask, decoder_input_ids=new_ids,
                        past_key_values=past, use_cache=use_cache, return_dict=True)
    else:
        outputs = model(input_ids=new_ids, attention_mask=attention_mask, past_key_values=past, use_cache=use_cache, return_dict=True)
    scores = outputs.logits[:, -1, :].float() # the lm may run in bf16
    if bad_words_ids is not None:
        for bad_word in bad_words_ids:
            assert len(bad_word) == 1
            if bad_word != [eos_token_id]:
                scores[:, bad_word[0]] = -float("inf")
    return scores, outputs.past_key_values


# hack of code from transformers/generation_utils.py
# to get our conditioning
def _generate_no_beam_search(
        model,
        conditioning_model,
        condition_lambda,
        precondition_topk,
        input_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        batch_size,
        attention_mask,
        use_cache,
        gen_start,
        model_kwargs,
        predictor_vocab_size=None,
        is_stop_token=None,
    ):
        """Generate sequences for each example without beam search (num_beams == 1).
        All returned sequence are generated independantly.
        gen_start: index in input_ids where the generated text starts; the conditioning model only sees tokens from there on.
        predictor_vocab_size: if given, lm tokens with ids from there on are never generated (the conditioning model can't embed them).
        is_stop_token: if given, function of a token id; a sequence also finishes after a token it's true for (e.g. a newline).
        """
        # length of generated sentences / unfinished sentences
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)

        past = None
        while cur_len < max_length:
            scores, past = _lm_next_token_scores(model, input_ids, past, attention_mask, model_kwargs, cur_len, min_length, max_length,
                                                 no_repeat_ngram_size, bad_words_ids, eos_token_id, repetition_penalty, batch_size, use_cache)
            if predictor_vocab_size is not None and scores.shape[1] > predictor_vocab_size:
                scores[:, predictor_vocab_size:] = -float("inf")

            top_logits, top_indices = scores.topk(precondition_topk, dim=1) # batch x topk
            tplus1_candidates = torch.cat([input_ids.unsqueeze(1).expand(-1, precondition_topk, -1), top_indices.unsqueeze(2)], dim=2)[:, :, gen_start:] # batch x topk x seq+1, with start token / prompt dropped
            expanded_lengths = torch.LongTensor([[cur_len + 1 - gen_start for _ in range(precondition_topk)] for _ in range(batch_size)]).to(scores.device)
            if condition_lambda == 0:
                condition_logits = torch.zeros_like(top_logits).float()
            else:
                condition_logits = conditioning_model(tplus1_candidates.flatten(0, 1), # batch*topk x seq+1
                                                    expanded_lengths.flatten(0, 1), # batch*topk
                                                    None,
                                                    None,
                                                    None)
                condition_logits = condition_logits.view(batch_size, precondition_topk, -1)[:, :, -1] # batch x topk of last formality pred
                condition_logits = condition_logits - torch.log(1 + torch.exp(condition_logits)) # get correct log probs
                # condition_logits = - torch.log(1 + torch.exp(condition_logits)) # for informal
            full_logits = top_logits + condition_lambda * condition_logits
            if do_sample:
                raise NotImplementedError
            else:
                # Greedy decoding
                next_token = top_indices[torch.arange(batch_size).to(top_indices.device), torch.argmax(full_logits, dim=-1)]

            # if do_sample:
            #     # Temperature (higher temperature => more likely to sample low probability tokens)
            #     if temperature != 1.0:
            #         scores = scores / temperature
            #     # Top-p/top-k filtering
            #     next_token_logscores = top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p)
            #     # Sample
            #     probs = F.softmax(next_token_logscores, dim=-1)
            #     next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            # else:
            #     # Greedy decoding
            #     next_token = torch.argmax(next_token_logits, dim=-1)

            # update generations and finished sentences
            if eos_token_id is not None:
                # pad finished sentences if eos_token_id exist
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token
            
            # add token and increase length by one
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            if eos_token_id is not None:
                eos_in_sents = tokens_to_add == eos_token_id
                # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
                # unfinished_sents is set to zero if eos in sentence
                unfinished_sents.mul_((~eos_in_sents).long())

            if is_stop_token is not None:
                stop_in_sents = torch.BoolTensor([is_stop_token(t) for t in tokens_to_add.tolist()]).to(unfinished_sents.device)
                unfinished_sents.mul_((~stop_in_sents).long())

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

            # extend attention_mask for new generated input if only decoder
            if model.config.is_encoder_decoder is False:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        return input_ids

if __name__=='__main__':
    parser = ArgumentParser()

    # DATA
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--dataset_info', type=str, required=True, help='saved dataset info')
    parser.add_argument('--model_string', type=str, default='Helsinki-NLP/opus-mt-es-en')
    parser.add_argument('--dtype', type=str, default=None, choices=['float32', 'bfloat16', 'float16'], help='precision to load the lm in (default: fp32); bfloat16 for the larger llms, in .venv-new')

    parser.add_argument('--input_text', type=str, default=None, required=True, help='text to run pred on')

    parser.add_argument('--precondition_topk', type=int, default=200, help='consider top k outputs from gpt at each step before conditioning and re-pruning')
    parser.add_argument('--do_sample', action='store_true', default=False, help='sample instead of greedy')
    parser.add_argument('--condition_lambda', type=float, default=1.0, help='lambda weight on conditioning model')
    parser.add_argument('--length_cutoff', type=int, default=512, help='max length')
    parser.add_argument('--prompt_template', type=str, default=FORMALITY_PROMPT_TEMPLATE, help='decoder-only lms only: prompt the input is put in at {input}; generation stops at the first newline. Ignored for encoder-decoder lms (marian)')

    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--debug', action='store_true', default=False)

    args = parse_args_with_config(parser)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)