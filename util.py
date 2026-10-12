import os
import time
import sys
import json
import resource
from contextlib import contextmanager

import torch

from constants import *

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


def save_checkpoint(state, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state, save_path)


def freeze(module):
    for param in module.parameters():
        param.requires_grad = False


def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_stats(model):
    """
    Parameter counts of a torch module, plus name/shape/count of each named parameter.
    """
    return {
        'num_params': sum(p.numel() for p in model.parameters()),
        'num_trainable_params': num_params(model),
        'parameters': [{'name': name, 'shape': list(p.shape), 'num_params': p.numel()} for name, p in model.named_parameters()],
    }


class ResourceTracker(object):
    """
    Measure the resources used by a run: wall-clock and cpu time, peak cpu memory and, on cuda, peak gpu memory.
    Wrap each part of the run in `with tracker.phase(name):` to also get its time and peak gpu memory.
    """
    def __init__(self, device):
        self.use_cuda = device == 'cuda' and torch.cuda.is_available()
        self.phases = {}
        self.gpu_peak_allocated = 0
        self.gpu_peak_reserved = 0
        if self.use_cuda:
            torch.cuda.reset_peak_memory_stats()
        self.start_wall = time.perf_counter()
        self.start_cpu = time.process_time()

    def _sync(self):
        if self.use_cuda:
            torch.cuda.synchronize() # cuda calls are async, wait for them so they're timed in the right phase

    def _update_gpu_peak(self):
        self.gpu_peak_allocated = max(self.gpu_peak_allocated, torch.cuda.max_memory_allocated())
        self.gpu_peak_reserved = max(self.gpu_peak_reserved, torch.cuda.max_memory_reserved())

    @contextmanager
    def phase(self, name):
        self._sync()
        if self.use_cuda:
            self._update_gpu_peak()
            torch.cuda.reset_peak_memory_stats() # so the peak below is this phase's
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            stats = {'time_s': time.perf_counter() - start}
            if self.use_cuda:
                stats['gpu_peak_allocated_mb'] = torch.cuda.max_memory_allocated() / 2**20
                stats['gpu_peak_reserved_mb'] = torch.cuda.max_memory_reserved() / 2**20
                self._update_gpu_peak()
            self.phases[name] = stats

    def summary(self, num_examples=None):
        self._sync()
        summary = {
            'total_time_s': time.perf_counter() - self.start_wall,
            'cpu_time_s': time.process_time() - self.start_cpu,
            'cpu_peak_memory_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**10, # ru_maxrss is in KB on linux
            'phases': self.phases,
        }
        if num_examples is not None:
            summary['num_examples'] = num_examples
            if num_examples > 0 and 'generation' in self.phases:
                summary['generation_time_per_example_s'] = self.phases['generation']['time_s'] / num_examples
        if self.use_cuda:
            self._update_gpu_peak()
            device = torch.cuda.current_device()
            summary['gpu'] = {
                'name': torch.cuda.get_device_name(device),
                'total_memory_mb': torch.cuda.get_device_properties(device).total_memory / 2**20,
                'peak_allocated_mb': self.gpu_peak_allocated / 2**20,
                'peak_reserved_mb': self.gpu_peak_reserved / 2**20,
            }
        else:
            summary['gpu'] = None
        return summary


def default_stats_file(task, model_name):
    """
    Run-specific default path for save_model_stats: logs/model_stats/<task>_<model>_<timestamp>.json
    """
    model_name = os.path.basename(model_name.rstrip('/')) # e.g. Helsinki-NLP/opus-mt-es-en -> opus-mt-es-en
    return os.path.join('logs', 'model_stats', '{}_{}_{}.json'.format(task, model_name, time.strftime('%Y%m%d-%H%M%S')))


def save_model_stats(save_path, model_string, model_path, language_model, conditioning_models, resources=None):
    """
    Save a json file describing the models involved in a FUDGE run:
    the base language model (name, parameter count) and the conditioning model(s), as a dict name -> (ckpt path, checkpoint, model).
    resources is optionally a ResourceTracker summary.
    """
    lm_stats = model_stats(language_model)
    stats = {
        'language_model': {
            'name': model_string,
            'model_path': model_path,
            'class': type(language_model).__name__,
            'num_params': lm_stats['num_params'],
        },
        'conditioning_models': {},
    }
    for name, (ckpt_path, checkpoint, model) in conditioning_models.items():
        stats['conditioning_models'][name] = {
            'ckpt': ckpt_path,
            'task': getattr(checkpoint['args'], 'task', None),
            'epoch': checkpoint.get('epoch'),
            **model_stats(model),
        }
    if resources is not None:
        stats['resources'] = resources
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(save_path, 'w') as wf:
        json.dump(stats, wf, indent=2)


def clamp(x, limit):
    return max(-limit, min(x, limit))


def pad_to_length(tensor, length, dim, value=0):
    """
    Pad tensor to given length in given dim using given value (value should be numeric)
    """
    assert tensor.size(dim) <= length
    if tensor.size(dim) < length:
        zeros_shape = list(tensor.shape)
        zeros_shape[dim] = length - tensor.size(dim)
        zeros_shape = tuple(zeros_shape)
        return torch.cat([tensor, torch.zeros(zeros_shape).type(tensor.type()).to(tensor.device).fill_(value)], dim=dim)
    else:
        return tensor


def pad_mask(lengths: torch.LongTensor) -> torch.ByteTensor:
    """
    Create a mask of seq x batch where seq = max(lengths), with 0 in padding locations and 1 otherwise. 
    """
    # lengths: bs. Ex: [2, 3, 1]
    max_seqlen = torch.max(lengths)
    expanded_lengths = lengths.unsqueeze(0).repeat((max_seqlen, 1))  # [[2, 3, 1], [2, 3, 1], [2, 3, 1]]
    indices = torch.arange(max_seqlen).unsqueeze(1).repeat((1, lengths.size(0))).to(lengths.device)  # [[0, 0, 0], [1, 1, 1], [2, 2, 2]]

    return expanded_lengths > indices  # pad locations are 0. #[[1, 1, 1], [1, 1, 0], [0, 1, 0]]. seqlen x bs


class ProgressMeter(object):
    """
    Display meter
    """
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries.append(time.ctime(time.time()))
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    """
    Display meter
    Computes and stores the average and current value
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)