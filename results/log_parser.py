"""
Collects the metrics of every run under logs/ into one csv, one row per (method, task, model), ready to plot
the chosen metric of each task against the language model's size.

logs/fudge/<task>/...: each run is a results_<tag>.log (output of eval_<task>_metrics.py) paired with the
    stats_<tag>.json written next to it (language model's name and number of parameters). The task is the first
    directory under logs/fudge (logs/fudge/formality/..., logs/fudge/toxicity/<run_id>/...).
logs/nado/<task>_<model>.log: each run is the stdout of a NADO run, whose last line points to the stats json in
    logs/nado/model_stats/ (language model and, once the run is complete, its metrics). The *.condor.log files are
    skipped. NADO metric names are renamed to the FUDGE ones (NADO_METRICS) so that both share the same columns.

Usage:
    python results/log_parser.py                                  # logs/fudge + logs/nado -> results/metrics.csv
    python results/log_parser.py --methods NADO --out results/nado.csv
"""
import argparse
import csv
import json
import os
import re

# metric of eval_<task>_metrics.py shown on the plot, for each task
PLOT_METRICS = {
    'formality': 'BLEU score',
    'toxicity': 'avg toxicity prob according to model',
    # 'hallucination': '<label printed by the hallucination eval script>',
}

# "<label> <number>" or "<label>: <number>" lines of a results log
METRIC_LINE = re.compile(r'^(?P<label>[^=].*?):?\s+(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)$')

# NADO metric name -> FUDGE label of the same metric (eval_<task>_metrics.py), unknown names are kept as they are
NADO_METRICS = {
    'avg_toxic_prob': 'avg toxicity prob according to model',
    'frac_toxic': 'fraction toxic according to model (prob >= 0.5)',
    'perplexity': 'perplexity under openai-community/gpt2-large',
    'bleu': 'BLEU score',
}

# logs/nado/<task>_<model>.log
NADO_LOG = re.compile(r'^(?P<task>[a-z]+)_(?P<model>.+)\.log$')
NADO_STATS_LINE = re.compile(r'^Saved run statistics to (?P<path>\S+\.json)$')
# NADO prints its "<name> <number>" metrics after this line (before it: training losses, epochs, ...)
NADO_METRICS_START = 'Saved generations to'

# what follows the model in a tag: _nontoxic_lambda1.0, _formal, ...
TAG_SUFFIX = re.compile(r'_(nontoxic|toxic|formal|informal)(_.*)?$')

FIELDS = ['method', 'task', 'model', 'lm_name', 'lm_params', 'lm_params_b', 'plot_metric', 'plot_value',
          'run', 'results_file', 'stats_file']


def parse_results(path):
    metrics = {}
    with open(path) as f:
        for line in f:
            match = METRIC_LINE.match(line.strip())
            if match:
                metrics[match.group('label')] = float(match.group('value'))
    return metrics


def find_stats(results_path):
    """stats_<tag>.json of results_<tag>.log, or, if missing, the one whose tag extends or is extended by <tag>
    (e.g. results_marianmt_formal.log <-> stats_marianmt_formality.json)"""
    run_dir = os.path.dirname(results_path)
    tag = os.path.basename(results_path)[len('results_'):-len('.log')]
    exact = os.path.join(run_dir, 'stats_{}.json'.format(tag))
    if os.path.exists(exact):
        return exact
    for name in sorted(os.listdir(run_dir)):
        if name.startswith('stats_') and name.endswith('.json'):
            stats_tag = name[len('stats_'):-len('.json')]
            if stats_tag.startswith(tag) or tag.startswith(stats_tag):
                return os.path.join(run_dir, name)
    return None


def make_row(method, task, model, run, results_path, stats_path, lm, metrics):
    plot_metric = PLOT_METRICS.get(task)
    if plot_metric not in metrics:
        print('WARNING: no "{}" in {}, skipping it (incomplete run?)'.format(plot_metric, results_path))
        return None
    row = {
        'method': method,
        'task': task,
        'model': model,
        'lm_name': lm['name'],
        'lm_params': lm['num_params'],
        'lm_params_b': lm['num_params'] / 1e9,
        'plot_metric': plot_metric,
        'plot_value': metrics[plot_metric],
        'run': run,
        'results_file': results_path,
        'stats_file': stats_path,
    }
    row.update(metrics)
    return row


def collect_fudge(logs_dir, method='FUDGE'):
    rows = []
    for dirpath, _, filenames in sorted(os.walk(logs_dir)):
        for name in sorted(filenames):
            if not (name.startswith('results_') and name.endswith('.log')):
                continue
            results_path = os.path.join(dirpath, name)
            rel_dir = os.path.relpath(dirpath, logs_dir)
            task = rel_dir.split(os.sep)[0]
            run = os.sep.join(rel_dir.split(os.sep)[1:])
            tag = name[len('results_'):-len('.log')]

            stats_path = find_stats(results_path)
            if stats_path is None:
                print('WARNING: no stats file for {}, skipping it'.format(results_path))
                continue
            with open(stats_path) as f:
                lm = json.load(f)['language_model']

            row = make_row(method, task, TAG_SUFFIX.sub('', tag), run, results_path, stats_path, lm,
                           parse_results(results_path))
            if row is not None:
                rows.append(row)
    return rows


def find_nado_stats(log_path, task, model):
    """stats json named in the "Saved run statistics to ..." line of the log (a path relative to the NADO repo,
    looked up by name in model_stats/ next to the log), or, if missing, the latest <task>_nado_<model>_*.json"""
    stats_dir = os.path.join(os.path.dirname(log_path), 'model_stats')
    with open(log_path) as f:
        for line in f:
            match = NADO_STATS_LINE.match(line.strip())
            if match:
                path = os.path.join(stats_dir, os.path.basename(match.group('path')))
                if os.path.exists(path):
                    return path
    if not os.path.isdir(stats_dir):
        return None
    prefix = '{}_nado_{}_'.format(task, model)
    candidates = sorted(n for n in os.listdir(stats_dir)
                        if n.startswith(prefix) and n.endswith('.json') and re.fullmatch(r'\d{8}-\d{6}', n[len(prefix):-len('.json')]))
    return os.path.join(stats_dir, candidates[-1]) if candidates else None


def parse_nado_log(path):
    """"<name> <number>" lines printed after the generations are saved"""
    metrics = {}
    started = False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(NADO_METRICS_START):
                started, metrics = True, {}  # keep only the last evaluation of the log
                continue
            match = METRIC_LINE.match(line) if started else None
            if match:
                metrics[match.group('label')] = float(match.group('value'))
    return metrics


def collect_nado(logs_dir, method='NADO'):
    rows = []
    for name in sorted(os.listdir(logs_dir)):
        match = NADO_LOG.match(name)
        if match is None or name.endswith('.condor.log'):
            continue
        log_path = os.path.join(logs_dir, name)
        task, model = match.group('task'), match.group('model')

        stats_path = find_nado_stats(log_path, task, model)
        if stats_path is None:
            print('WARNING: no stats file for {}, skipping it'.format(log_path))
            continue
        with open(stats_path) as f:
            stats = json.load(f)

        # metrics saved in the stats json, or, for older runs, printed in the log
        nado_metrics = stats.get('metrics') or parse_nado_log(log_path)
        metrics = {NADO_METRICS.get(k, k): float(v) for k, v in nado_metrics.items()}
        row = make_row(method, task, model, '', log_path, stats_path, stats['language_model'], metrics)
        if row is not None:
            rows.append(row)
    return rows


# method -> (subdir of the logs dir, collector)
METHODS = {
    'FUDGE': ('fudge', collect_fudge),
    'NADO': ('nado', collect_nado),
}


def main():
    parser = argparse.ArgumentParser(description='collect the metrics of the runs in a logs dir into one csv')
    parser.add_argument('--logs_dir', type=str, default='logs', help='dir with a fudge/ and a nado/ subdir to scan')
    parser.add_argument('--methods', nargs='+', default=list(METHODS), choices=list(METHODS), help='methods to collect')
    parser.add_argument('--out', type=str, default='results/metrics.csv', help='csv to write')
    parser.add_argument('--append', action='store_true', help='add the rows to an existing --out')
    args = parser.parse_args()

    rows = []
    for method in args.methods:
        subdir, collect = METHODS[method]
        method_dir = os.path.join(args.logs_dir, subdir)
        if not os.path.isdir(method_dir):
            print('WARNING: no {} dir, skipping {}'.format(method_dir, method))
            continue
        rows += collect(method_dir, method)
    if args.append and os.path.exists(args.out):
        with open(args.out) as f:
            rows = list(csv.DictReader(f)) + rows

    metric_fields = sorted({k for row in rows for k in row} - set(FIELDS))
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS + metric_fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r['method'], r['task'], float(r['lm_params']))))
    print('{} rows written to {}'.format(len(rows), args.out))


if __name__ == '__main__':
    main()
