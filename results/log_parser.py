"""
Collects the metrics of every run under logs/ into one csv, one row per (method, task, model), ready to plot
the chosen metric of each task against the language model's size.

Each run is a results_<tag>.log (output of eval_<task>_metrics.py) paired with the stats_<tag>.json written
next to it (language model's name and number of parameters). The task is the first directory under the logs dir
(logs/formality/..., logs/toxicity/<run_id>/...).

Usage:
    python results/log_parser.py                                  # FUDGE logs in logs/ -> results/metrics.csv
    python results/log_parser.py --logs_dir ../nado/logs --method NADO --out results/metrics.csv --append
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


def collect(logs_dir, method):
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

            metrics = parse_results(results_path)
            plot_metric = PLOT_METRICS.get(task)
            if plot_metric not in metrics:
                print('WARNING: no "{}" in {}, skipping it (incomplete run?)'.format(plot_metric, results_path))
                continue

            row = {
                'method': method,
                'task': task,
                'model': TAG_SUFFIX.sub('', tag),
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
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description='collect the metrics of the runs in a logs dir into one csv')
    parser.add_argument('--logs_dir', type=str, default='logs', help='logs/<task>/... dir to scan')
    parser.add_argument('--method', type=str, default='FUDGE', help='name of the method that produced the logs')
    parser.add_argument('--out', type=str, default='results/metrics.csv', help='csv to write')
    parser.add_argument('--append', action='store_true', help='add the rows to an existing --out (e.g. a second method)')
    args = parser.parse_args()

    rows = collect(args.logs_dir, args.method)
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
