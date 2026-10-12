#!/bin/bash
set -e

cd /lustrehome/llaraspata/fudge

# venv of the model entry in config.yaml: .venv-new for the newer llm families (llama, qwen, gemma), else .venv
VENV=$(.venv/bin/python -c "import yaml,sys; print(yaml.safe_load(open('config.yaml'))['models'][sys.argv[1]].get('venv') or '.venv')" "$4")
source "$VENV/bin/activate"

nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.cuda.get_arch_list())"
export CUDA_LAUNCH_BLOCKING=1   # surfaces the real CUDA error at the failing call

# $1 ckpt, $2 dataset_info, $3 source file, $4 model entry in config.yaml, $5 generations output,
# $6 bleu reference, $7 evaluator ckpt, $8 evaluator dataset_info, $9 metrics output
python -u evaluate_formality.py --ckpt "$1" --dataset_info "$2" --in_file "$3" --model "$4" > "$5"

python eval_formality_metrics.py --pred "$5" --ref "$6" --ckpt "$7" --dataset_info "$8" > "$9" # the evaluator's tokenizer is read from its ckpt