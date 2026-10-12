#!/usr/bin/env bash
# Runs the toxicity task locally (no recas/HTCondor) with every newer llm of config.yaml (Llama 3.2, Gemma 4, Qwen2.5):
#   1. creates .venv-new (Python 3.11, requirements-new.txt) and installs everything, if needed
#   2. downloads the models into the huggingface cache. Llama and Gemma are gated: accept their license on huggingface
#      first and set HF_TOKEN (or run `hf auth login`); models that can't be downloaded are skipped
#   3. downloads and preprocesses RealToxicityPrompts, if train_data/RealToxicityPrompts/processed isn't there yet
#   4. trains one toxicity predictor per family (models of a family share the tokenizer) into ckpt/toxicity/<family>/;
#      an existing ckpt/toxicity/<family>/model_best.pth.tar is reused, delete it to retrain
#   5. for each model: generates continuations of the test prompts with FUDGE, then computes the metrics
#
# Usage: ./run_toxicity_new_llms.sh
#
# Everything of a run goes to logs/toxicity/<RUN_ID>/:
#   run.log                                     the whole output of this script
#   train_predictor_<family>.log                predictor training (only for families trained in this run)
#   preds_<model>_<target>_lambda<L>.log        generations, one per line, aligned with the prompts
#   results_<model>_<target>_lambda<L>.log      metrics of those generations (eval_toxicity_metrics.py)
#   stats_<model>_<target>_lambda<L>.json       model statistics and resources used for generation
#   errors_<model>_<target>_lambda<L>.log       stderr of generation and metrics
#   summary.txt                                 which models succeeded, were skipped or failed
#
# Environment variables:
#   MODELS            config.yaml model entries to run (default: all 7 newer llms)
#   NUM_PROMPTS       only use the first N test prompts (default: all ~20k; generation is one prompt at a time,
#                     so expect hours per model)
#   TARGET            nontoxic (default) or toxic: label to steer generation towards
#   CONDITION_LAMBDA  weight of the predictor (default: 1.0; 0 = plain greedy decoding, no FUDGE)
#   EPOCHS            predictor training epochs (default: 20)
#   NUM_WORKERS       data loader workers for training (default: 20)
#   RUN_ID            name of the run's log dir (default: current date and time)
#   CUDA_VISIBLE_DEVICES  gpu to use (default: 0)
#   VENV_DIR          venv to create/use (default: .venv-new)
#   TORCH_INDEX       index for torch (default: https://download.pytorch.org/whl/cu126, CUDA 12.6 builds)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODELS="${MODELS:-llama32_1b llama32_3b gemma4_e2b gemma4_e4b qwen25_0_5b qwen25_1_5b qwen25_3b}"
NUM_PROMPTS="${NUM_PROMPTS:-}"
TARGET="${TARGET:-nontoxic}"
CONDITION_LAMBDA="${CONDITION_LAMBDA:-1.0}"
EPOCHS="${EPOCHS:-20}"
NUM_WORKERS="${NUM_WORKERS:-20}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
VENV_DIR="${VENV_DIR:-.venv-new}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA_DIR=train_data/RealToxicityPrompts/processed
LOG_DIR="logs/toxicity/$RUN_ID"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/run.log") 2>&1

echo "== run $RUN_ID: models [$MODELS], target $TARGET, lambda $CONDITION_LAMBDA, prompts ${NUM_PROMPTS:-all}, gpu $CUDA_VISIBLE_DEVICES"
echo "== logs in $LOG_DIR"

# ----------
# 1. VENV
# ----------
if ! command -v uv >/dev/null; then
    echo "== uv not found, installing it (https://docs.astral.sh/uv/)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "== setting up $VENV_DIR"
[ -x "$VENV_DIR/bin/python" ] || uv venv --python 3.11 "$VENV_DIR"
uv pip install --python "$VENV_DIR/bin/python" -r requirements-new.txt \
    --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
source "$VENV_DIR/bin/activate"
export PYTHONWARNINGS="ignore::UserWarning:pronouncing" # pkg_resources deprecation, on every import of data.py
python -c "import torch, transformers; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '| transformers', transformers.__version__)"
python -c "import torch; assert torch.cuda.is_available(), 'no gpu available'"

if [ -z "${HF_TOKEN:-}" ] && [ ! -f "${HF_HOME:-$HOME/.cache/huggingface}/token" ]; then
    echo "WARNING: no huggingface token (HF_TOKEN or \`hf auth login\`): the gated llama and gemma models will be skipped"
fi

# ----------
# 2. MODELS
# ----------
model_string() { # model_string <config.yaml model entry>
    python -c "import yaml,sys; print(yaml.safe_load(open('config.yaml'))['models'][sys.argv[1]]['model_string'])" "$1"
}

declare -A STATUS # model entry -> result, for the summary
AVAILABLE=()
for model in $MODELS; do
    name=$(model_string "$model")
    echo "== downloading $model ($name)"
    if hf download "$name" --exclude "original/*" --quiet >/dev/null; then # original/: llama's duplicate weights in meta's format
        AVAILABLE+=("$model")
    else
        echo "WARNING: couldn't download $name, skipping $model. If it is gated (llama, gemma), accept its license on huggingface and set HF_TOKEN"
        STATUS[$model]="skipped (download failed)"
    fi
done

# ----------
# 3. DATA
# ----------
if [ ! -f "$DATA_DIR/test_prompts" ]; then
    echo "== preparing RealToxicityPrompts"
    mkdir -p train_data/RealToxicityPrompts
    [ -f train_data/RealToxicityPrompts/prompts.jsonl ] || curl -fL --retry 3 -o train_data/RealToxicityPrompts/prompts.jsonl \
        https://huggingface.co/datasets/allenai/real-toxicity-prompts/resolve/main/prompts.jsonl
    python prepare_toxicity_data.py --in_file train_data/RealToxicityPrompts/prompts.jsonl --out_dir "$DATA_DIR"
fi

PROMPTS="$DATA_DIR/test_prompts"
if [ -n "$NUM_PROMPTS" ]; then
    PROMPTS="$LOG_DIR/test_prompts_first$NUM_PROMPTS"
    head -n "$NUM_PROMPTS" "$DATA_DIR/test_prompts" > "$PROMPTS"
fi
echo "== $(wc -l < "$PROMPTS") test prompts from $PROMPTS"

# ----------
# 4. PREDICTORS, one per family
# ----------
family() { # llama32_1b -> llama32, gemma4_e2b -> gemma4, qwen25_0_5b -> qwen25
    echo "${1%%_*}"
}

declare -A PREDICTOR_OK # family -> 1 if its predictor is there
for model in "${AVAILABLE[@]}"; do
    fam=$(family "$model")
    [ -n "${PREDICTOR_OK[$fam]:-}" ] && continue
    ckpt_dir="ckpt/toxicity/$fam"
    if [ -f "$ckpt_dir/model_best.pth.tar" ]; then
        echo "== reusing the $fam predictor in $ckpt_dir"
        PREDICTOR_OK[$fam]=1
        continue
    fi
    echo "== training the $fam predictor (with the $model tokenizer) into $ckpt_dir, log in $LOG_DIR/train_predictor_$fam.log"
    if python -u main.py --dataset rtp --model "$model" --save_dir "$ckpt_dir" --num_workers "$NUM_WORKERS" \
            --batch_size 32 --validation_freq 1 --lr 2e-5 --epochs "$EPOCHS" > "$LOG_DIR/train_predictor_$fam.log" 2>&1; then
        PREDICTOR_OK[$fam]=1
    else
        echo "ERROR: training the $fam predictor failed, see $LOG_DIR/train_predictor_$fam.log"
        PREDICTOR_OK[$fam]=0
    fi
done

# ----------
# 5. GENERATION AND METRICS
# ----------
for model in "${AVAILABLE[@]}"; do
    fam=$(family "$model")
    ckpt_dir="ckpt/toxicity/$fam"
    if [ "${PREDICTOR_OK[$fam]}" != 1 ]; then
        STATUS[$model]="skipped (no $fam predictor)"
        continue
    fi
    tag="${model}_${TARGET}_lambda${CONDITION_LAMBDA}"
    preds="$LOG_DIR/preds_$tag.log"
    results="$LOG_DIR/results_$tag.log"
    errors="$LOG_DIR/errors_$tag.log"

    echo "== $model: generating into $preds"
    if ! python -u evaluate_toxicity.py --dataset rtp --model "$model" \
            --ckpt "$ckpt_dir/model_best.pth.tar" --dataset_info "$ckpt_dir/dataset_info" --in_file "$PROMPTS" \
            --target "$TARGET" --condition_lambda "$CONDITION_LAMBDA" --stats_file "$LOG_DIR/stats_$tag.json" \
            > "$preds" 2> "$errors"; then
        echo "ERROR: generation with $model failed, see $errors"
        STATUS[$model]="failed (generation)"
        continue
    fi

    echo "== $model: metrics into $results"
    if ! python eval_toxicity_metrics.py --pred "$preds" --prompts "$PROMPTS" \
            --ckpt "$ckpt_dir/model_best.pth.tar" --dataset_info "$ckpt_dir/dataset_info" \
            > "$results" 2>> "$errors"; then
        echo "ERROR: metrics of $model failed, see $errors"
        STATUS[$model]="failed (metrics)"
        continue
    fi
    STATUS[$model]="done"
    cat "$results"
done

# ----------
# SUMMARY
# ----------
{
    echo "run $RUN_ID: target $TARGET, lambda $CONDITION_LAMBDA, $(wc -l < "$PROMPTS") prompts from $PROMPTS"
    for model in $MODELS; do
        printf '%-14s %s\n' "$model" "${STATUS[$model]:-not run}"
    done
} | tee "$LOG_DIR/summary.txt"
