import os

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

from constants import *

def hf_hub_cache():
    """
    The huggingface hub cache, where setup.sh downloads models: $HF_HUB_CACHE, else $HF_HOME/hub, else ~/.cache/huggingface/hub.
    """
    if os.environ.get('HF_HUB_CACHE'):
        return os.path.expanduser(os.environ['HF_HUB_CACHE'])
    hf_home = os.environ.get('HF_HOME') or os.path.join(os.environ.get('XDG_CACHE_HOME') or '~/.cache', 'huggingface')
    return os.path.join(os.path.expanduser(hf_home), 'hub')


def resolve_model_string(model_string):
    """
    Use the copy of model_string in the huggingface hub cache if there is one (downloaded by setup.sh):
    transformers 3.4 can't download from the hub anymore, nor read the hub cache by itself.
    Local paths and models not in the cache are returned unchanged.
    """
    if os.path.isdir(model_string):
        return model_string
    repo_dir = os.path.join(hf_hub_cache(), 'models--' + model_string.replace('/', '--'))
    ref_file = os.path.join(repo_dir, 'refs', 'main')
    if os.path.isfile(ref_file):
        with open(ref_file, 'r') as rf:
            snapshot_dir = os.path.join(repo_dir, 'snapshots', rf.read().strip())
        if os.path.isdir(snapshot_dir):
            return snapshot_dir
    return model_string


def load_tokenizer(model_string):
    """
    Load the tokenizer for model_string (AutoTokenizer picks the right class, e.g. MarianTokenizer, GPT2TokenizerFast).
    Adds PAD_TOKEN, as the predictors expect. Returns (tokenizer, pad_id).
    """
    tokenizer = AutoTokenizer.from_pretrained(resolve_model_string(model_string))
    tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
    pad_id = tokenizer.encode(PAD_TOKEN)[0] # actually just the vocab size
    return tokenizer, pad_id


def load_lm(model_string, device='cuda', model_path=None, **kwargs):
    """
    Load the language model for model_string in eval mode: encoder-decoder models (e.g. Marian) with
    AutoModelForSeq2SeqLM, decoder-only ones (e.g. GPT-2) with AutoModelForCausalLM.
    model_path optionally points to finetuned weights (a .ckpt file, or a dir containing one) loaded on top.
    kwargs are passed to from_pretrained.
    """
    model_string = resolve_model_string(model_string)
    config = AutoConfig.from_pretrained(model_string)
    model_class = AutoModelForSeq2SeqLM if config.is_encoder_decoder else AutoModelForCausalLM
    with torch.no_grad(): # transformers 3.4's Marian/Bart positional embedding init fails on torch >= 1.8 otherwise
        model = model_class.from_pretrained(model_string, **kwargs)

    if model_path is not None:
        if os.path.isdir(model_path):
            for _, _, files in os.walk(model_path):
                for fname in files:
                    if fname.endswith('.ckpt'):
                        model_path = os.path.join(model_path, fname)
                        break
        ckpt = torch.load(model_path, map_location='cpu')
        try:
            model.load_state_dict(ckpt['state_dict'])
        except:
            state_dict = {}
            for key in ckpt['state_dict'].keys():
                assert key.startswith('model.')
                state_dict[key[6:]] = ckpt['state_dict'][key]
            model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    return model


def load_tokenizer_and_lm(model_string, device='cuda', model_path=None, **kwargs):
    """
    Common entry point: returns (tokenizer, pad_id, model).
    """
    tokenizer, pad_id = load_tokenizer(model_string)
    model = load_lm(model_string, device=device, model_path=model_path, **kwargs)
    return tokenizer, pad_id, model
