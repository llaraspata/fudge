#!/bin/bash
set -e

cd /lustrehome/llaraspata/fudge

source .venv/bin/activate

nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.cuda.get_arch_list())"
export CUDA_LAUNCH_BLOCKING=1   # surfaces the real CUDA error at the failing call

# $1 ckpt, $2 dataset_info, $3 prompts file, $4 model entry in config.yaml, $5 generations output, $6 metrics output
python -u evaluate_toxicity.py --dataset rtp --ckpt "$1" --dataset_info "$2" --in_file "$3" --model "$4" > "$5"

MODEL_STRING=$(python -c "import yaml,sys; print(yaml.safe_load(open('config.yaml'))['models'][sys.argv[1]]['model_string'])" "$4")
python eval_toxicity_metrics.py --pred "$5" --prompts "$3" --ckpt "$1" --dataset_info "$2" --model_string "$MODEL_STRING" > "$6"
