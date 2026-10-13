"""
Plots the chosen metric of each task against the language model's size, one point per (task, model) row of the
csv written by log_parser.py for the given method. Color identifies the task.

Since the metrics live on different scales (BLEU is ~0-100, the toxicity prob is 0-1), the formality task is read
on the left y axis and the other tasks (toxicity, hallucination) on the right one.

Generation is steered toward nontoxic text, so toxicity is plotted as avg non-toxic prob = 1 - avg toxicity prob
(higher is better, like BLEU).

Usage:
    python results/plot_metrics.py                                    # results/metrics.csv -> results/metrics.png
    python results/plot_metrics.py --csv results/metrics.csv --out results/metrics.pdf --labels
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

TASK_ORDER = ['formality', 'toxicity', 'hallucination']
# tasks take colors in this fixed order, so a task keeps its color whichever others are in the csv
TASK_COLORS = {'formality': '#2a78d6', 'toxicity': '#eb6834', 'hallucination': '#1baf7a'}
LEFT_AXIS_TASKS = {'formality'}

# metric of a task shown on the plot, if it is not plot_value as is: task -> (label, transform)
PLOT_TRANSFORMS = {
    'toxicity': ('avg non-toxic prob', lambda v: 1 - v),
}

TEXT = '#1f1f1e'
MUTED = '#6b6a64'
GRID = '#e4e3dd'


def read_rows(path, method):
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r.get('plot_value') and r['method'] == method]
    for r in rows:
        r['lm_params_b'] = float(r['lm_params_b'])
        r['plot_value'] = float(r['plot_value'])
        if r['task'] in PLOT_TRANSFORMS:
            label, transform = PLOT_TRANSFORMS[r['task']]
            r['plot_metric'], r['plot_value'] = label, transform(r['plot_value'])
    return rows


def params_label(x, _):
    return f'{x:g}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=os.path.join(os.path.dirname(__file__), 'metrics.csv'))
    parser.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'metrics.png'))
    parser.add_argument('--method', default='FUDGE', help='rows of the csv to plot')
    parser.add_argument('--title', default="Baselines' performance")
    parser.add_argument('--labels', action='store_true', help='write the model name next to each point', default=True)
    parser.add_argument('--linear_x', action='store_true', help='linear instead of log scale for the params')
    args = parser.parse_args()

    rows = read_rows(args.csv, args.method)
    tasks = sorted({r['task'] for r in rows}, key=lambda t: (TASK_ORDER + [t]).index(t))
    colors = {t: TASK_COLORS.get(t, MUTED) for t in tasks}

    fig, left = plt.subplots(figsize=(8.5, 4.5))
    right = left.twinx()
    axis_tasks = {left: [t for t in tasks if t in LEFT_AXIS_TASKS], right: [t for t in tasks if t not in LEFT_AXIS_TASKS]}

    for task in tasks:
        ax = left if task in LEFT_AXIS_TASKS else right
        pts = sorted((r for r in rows if r['task'] == task), key=lambda r: r['lm_params_b'])
        ax.scatter([r['lm_params_b'] for r in pts], [r['plot_value'] for r in pts],
                   s=80, color=colors[task], edgecolor='white', linewidth=1.5, zorder=3)
        if args.labels:
            for r in pts:
                ax.annotate(r['lm_name'].split('/')[-1], (r['lm_params_b'], r['plot_value']),
                            xytext=(0, 9), textcoords='offset points', ha='center', fontsize=10, color=MUTED)

    # each y label names its tasks and their metrics, e.g. "Formality: BLEU score"
    for ax, side in ((left, 'left'), (right, 'right')):
        ax_tasks = axis_tasks[ax]
        if not ax_tasks:
            ax.set_yticks([])
            ax.spines[side].set_visible(False)
            continue
        label = '\n'.join(f"{t.capitalize()}: {' / '.join(sorted({r['plot_metric'] for r in rows if r['task'] == t}))}"
                          for t in ax_tasks)
        ax.set_ylabel(label, color=MUTED, fontsize=14)
        ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.margins(y=0.15)

    left.grid(axis='x', color=GRID, linestyle=':', linewidth=1)
    left.set_axisbelow(True)
    left.spines['bottom'].set_color(MUTED)
    for ax in (left, right):
        ax.spines['top'].set_visible(False)
    left.spines['right'].set_visible(False)
    right.spines['left'].set_visible(False)
    right.spines['bottom'].set_visible(False)

    if not args.linear_x:
        left.set_xscale('log')
    left.xaxis.set_major_formatter(FuncFormatter(params_label))
    left.xaxis.set_minor_formatter(FuncFormatter(lambda x, _: ''))

    left.set_title(args.title, color=TEXT, fontsize=20, pad=14)
    left.set_xlabel("Language Model's params (billions)", color=TEXT, fontsize=14)

    handles = [Line2D([], [], linestyle='none', marker='o', markersize=9, color=c, label=t.capitalize())
               for t, c in colors.items()]
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False, labelcolor=TEXT)

    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches='tight')
    print(f'saved {args.out}')
    pdf_out = os.path.splitext(args.out)[0] + '.pdf'
    fig.savefig(pdf_out, bbox_inches='tight')
    print(f'saved {pdf_out}')


if __name__ == '__main__':
    main()
