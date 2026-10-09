#!/bin/bash
set -e

cd /lustrehome/llaraspata/fudge

source .venv/bin/activate

python -u evaluate_formality.py --ckpt "$1" --dataset_info "$2" --in_file "$3" --model "$4" > "$5"

python eval_formality_metrics.py --pred "$5" --ref "$6" --ckpt "$7" --dataset_info "$8"  > "$9"