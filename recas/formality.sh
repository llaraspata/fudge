#!/bin/bash
set -e

cd /lustrehome/llaraspata/fudge

source .venv/bin/activate

nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.cuda.get_arch_list())"
export CUDA_LAUNCH_BLOCKING=1   # surfaces the real CUDA error at the failing call


python -u evaluate_formality.py --ckpt "$1" --dataset_info "$2" --in_file "$3" --model "$4" > "$5"

python eval_formality_metrics.py --pred "$5" --ref "$6" --ckpt "$7" --dataset_info "$8"  > "$9"