import os
import pickle
import types

import torch
import transformers
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
    Local paths and models not in the cache are returned unchanged, as is everything with recent transformers (.venv-new),
    which reads the cache itself and downloads missing files.
    """
    if os.path.isdir(model_string) or int(transformers.__version__.split('.')[0]) >= 4:
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
    pad_id = tokenizer.convert_tokens_to_ids(PAD_TOKEN) # actually just the vocab size. not encode(PAD_TOKEN)[0]: llama/gemma tokenizers prepend bos
    return tokenizer, pad_id


def encode_for_predictor(tokenizer, text, return_tensors=None):
    """
    Encode text as the predictors see it: like tokenizer.encode, but without the bos token llama/gemma tokenizers prepend,
    since at generation time the predictor only sees the generated tokens. Other special tokens (e.g. marian's eos) are kept
    as before, so predictors trained with gpt2/marian are unaffected.
    """
    ids = tokenizer.encode(text)
    if tokenizer.bos_token_id is not None and len(ids) > 0 and ids[0] == tokenizer.bos_token_id:
        ids = ids[1:]
    if return_tensors == 'pt':
        return torch.LongTensor([ids])
    return ids


class _MissingClass:
    """
    Stand-in for a pickled class whose module is not installed (e.g. pytorch_lightning callbacks in the Marian .ckpt).
    """
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


class _LenientUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            return _MissingClass


_lenient_pickle = types.SimpleNamespace(Unpickler=_LenientUnpickler, load=lambda f, **kw: _LenientUnpickler(f, **kw).load(), __name__='pickle')


def load_checkpoint(path, map_location='cpu'):
    """
    torch.load for our checkpoints, which pickle the training args: torch >= 2.6 only loads plain weights by default.
    Classes from packages that are not installed (e.g. pytorch_lightning) are replaced with placeholders, only the weights matter.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False, pickle_module=_lenient_pickle)
    except TypeError: # torch < 1.13 has no weights_only
        return torch.load(path, map_location=map_location, pickle_module=_lenient_pickle)


def load_lm(model_string, device='cuda', model_path=None, dtype=None, **kwargs):
    """
    Load the language model for model_string in eval mode: encoder-decoder models (e.g. Marian) with
    AutoModelForSeq2SeqLM, decoder-only ones (e.g. GPT-2) with AutoModelForCausalLM.
    model_path optionally points to finetuned weights (a .ckpt file, or a dir containing one) loaded on top.
    dtype optionally loads the weights in another precision (e.g. 'bfloat16', for the larger llms); None keeps fp32.
    kwargs are passed to from_pretrained.
    """
    model_string = resolve_model_string(model_string)
    if dtype is not None:
        kwargs['dtype' if int(transformers.__version__.split('.')[0]) >= 5 else 'torch_dtype'] = getattr(torch, dtype)
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
        ckpt = load_checkpoint(model_path)
        try:
            model.load_state_dict(ckpt['state_dict'])
        except:
            state_dict = {}
            for key in ckpt['state_dict'].keys():
                assert key.startswith('model.')
                state_dict[key[6:]] = ckpt['state_dict'][key]
            # checkpoints from older transformers have no lm_head.weight: it is tied to the input embeddings
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            tied = set(getattr(model, '_tied_weights_keys', None) or []) | {'lm_head.weight'}
            if set(missing) - tied or unexpected:
                raise RuntimeError('error loading {}: missing keys {}, unexpected keys {}'.format(model_path, missing, unexpected))
            model.tie_weights()

    model = model.to(device)
    model.eval()
    return model


def load_tokenizer_and_lm(model_string, device='cuda', model_path=None, dtype=None, **kwargs):
    """
    Common entry point: returns (tokenizer, pad_id, model).
    """
    tokenizer, pad_id = load_tokenizer(model_string)
    model = load_lm(model_string, device=device, model_path=model_path, dtype=dtype, **kwargs)
    return tokenizer, pad_id, model
