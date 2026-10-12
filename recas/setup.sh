#!/usr/bin/env bash
# Sets up FUDGE from scratch:
#   1. a Python 3.8 venv (.venv) with the pinned requirements, made with uv
#   2. the huggingface models used by the scripts, into the huggingface cache (transformers 3.4 can't download them itself anymore)
#   3. the pretrained predictor checkpoints and training data (large_files.zip, ~7.5GB) into ckpt/, train_data/, topic_human_evals/
#   4. with --new: a second, Python 3.11 venv (.venv-new) with the recent stack of requirements-new.txt, and the newer llm
#      families of config.yaml (Llama 3.2, Gemma 4, Qwen2.5) into the huggingface cache. Llama and Gemma are gated:
#      accept their license on huggingface first and set HF_TOKEN (or run `hf auth login`)
#
# Usage: ./setup.sh [--skip-venv] [--skip-models] [--skip-data] [--new]
# Everything already in place is skipped, so it's safe to re-run.
#
# Environment variables:
#   VENV_DIR      where to create the venv (default: .venv)
#   TORCH_SPEC    torch to install over the pinned 1.7.0 (default: torch==2.4.1, built for Ampere/Ada/Hopper GPUs up to sm_90;
#                 it is the last release for python 3.8); set it to empty to keep torch==1.7.0
#   TORCH_INDEX   index for TORCH_SPEC (default: https://download.pytorch.org/whl/cu121)
#   HF_HOME       huggingface root dir; models go to $HF_HOME/hub (default: ~/.cache/huggingface/hub)
#   HF_HUB_CACHE  huggingface model cache, overrides $HF_HOME/hub. lm.py reads models from the same place,
#                 so set the same value when running the scripts
#   HF_TOKEN      optional huggingface token, used for model downloads (needed for the gated ones of --new)
#   NEW_VENV_DIR  where to create the venv of --new (default: .venv-new)
#   NEW_TORCH_INDEX  index for the torch of --new (default: https://download.pytorch.org/whl/cu126, CUDA 12.6 builds)

set -euo pipefail

SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
ROOT="$(dirname "$(dirname "$SCRIPT")")" # the repo root, this script is in recas/
cd "$ROOT"

VENV_DIR="${VENV_DIR:-.venv}"
TORCH_SPEC="${TORCH_SPEC-torch==2.4.1}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
DATA_URL="https://naacl2021-fudge-files.s3.amazonaws.com/large_files.zip"
NEW_VENV_DIR="${NEW_VENV_DIR:-.venv-new}"
NEW_TORCH_INDEX="${NEW_TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"

DO_VENV=1; DO_MODELS=1; DO_DATA=1; DO_NEW=0
for arg in "$@"; do
    case "$arg" in
        --skip-venv) DO_VENV=0 ;;
        --skip-models) DO_MODELS=0 ;;
        --skip-data) DO_DATA=0 ;;
        --new) DO_NEW=1 ;;
        -h|--help) sed -n '2,24p' "$SCRIPT" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $arg (see --help)" >&2; exit 1 ;;
    esac
done

# ----------
# VENV
# ----------
if ! command -v uv >/dev/null; then
    echo "uv not found, installing it (https://docs.astral.sh/uv/)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ "$DO_VENV" = 1 ]; then
    echo "== creating venv in $VENV_DIR"
    [ -x "$VENV_DIR/bin/python" ] || uv venv --python 3.8 "$VENV_DIR"
    PY="$VENV_DIR/bin/python"

    uv pip install --python "$PY" -r requirements.txt
    # pytorch-lightning pulls in tensorboard, which otherwise brings protobuf 5.x and breaks the import
    uv pip install --python "$PY" "protobuf==3.20.3"
    if [ -n "$TORCH_SPEC" ]; then
        uv pip install --python "$PY" "$TORCH_SPEC" \
            --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
    fi

    "$PY" -c "import torch, transformers, pytorch_lightning, yaml; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '| transformers', transformers.__version__)"
fi

# ----------
# MODELS
# ----------
# download_model <name> <files...>
# Downloads with the huggingface hub cli (run through uvx, it needs a newer python than the venv) into the hub cache,
# where lm.resolve_model_string finds it. Use the current repo names (e.g. openai-community/gpt2-medium, not gpt2-medium):
# the hub doesn't redirect old names for the weights. Only files transformers 3.4 can read are fetched.
download_model() {
    local name="$1"; shift
    echo "== $name"
    uvx --from huggingface_hub hf download "$name" "$@" >/dev/null
}

if [ "$DO_MODELS" = 1 ]; then
    # base models (config.yaml)
    download_model Helsinki-NLP/opus-mt-es-en \
        config.json pytorch_model.bin source.spm target.spm vocab.json tokenizer_config.json
    download_model openai-community/gpt2-medium config.json pytorch_model.bin vocab.json merges.txt
    download_model openai-community/gpt2-large config.json pytorch_model.bin vocab.json merges.txt

    # evaluation models (eval_topic_metrics.py, eval_poetry_metrics.py)
    download_model openai-community/openai-gpt config.json pytorch_model.bin vocab.json merges.txt
    download_model transfo-xl/transfo-xl-wt103 config.json pytorch_model.bin vocab.pkl
    download_model textattack/roberta-base-CoLA \
        config.json pytorch_model.bin vocab.json merges.txt special_tokens_map.json tokenizer_config.json
fi

# ----------
# CHECKPOINTS AND TRAINING DATA
# ----------
if [ "$DO_DATA" = 1 ]; then
    if [ -d ckpt/formality ] && [ -d train_data ]; then
        echo "== ckpt/ and train_data/ already there, skipping large_files.zip"
    else
        echo "== downloading large_files.zip (~7.5GB)"
        # -C - resumes a partial download
        curl -fL --retry 3 -C - -o large_files.zip "$DATA_URL"
        unzip -o -q large_files.zip -d "$ROOT"
        rm large_files.zip
    fi
fi

# ----------
# NEW LLM FAMILIES (--new)
# ----------
if [ "$DO_NEW" = 1 ]; then
    echo "== creating venv in $NEW_VENV_DIR"
    [ -x "$NEW_VENV_DIR/bin/python" ] || uv venv --python 3.11 "$NEW_VENV_DIR"
    NEW_PY="$NEW_VENV_DIR/bin/python"
    uv pip install --python "$NEW_PY" -r requirements-new.txt \
        --index-url "$NEW_TORCH_INDEX" --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
    "$NEW_PY" -c "import torch, transformers; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '| transformers', transformers.__version__)"

    if [ "$DO_MODELS" = 1 ]; then
        # the model_string of every config.yaml entry using this venv. Recent transformers read the hub cache themselves.
        for name in $("$NEW_PY" -c "import yaml; print(' '.join(m['model_string'] for m in yaml.safe_load(open('config.yaml'))['models'].values() if m.get('venv') == '.venv-new'))"); do
            echo "== $name"
            # original/: llama's duplicate weights in meta's format
            "$NEW_VENV_DIR/bin/hf" download "$name" --exclude "original/*" >/dev/null \
                || echo "WARNING: couldn't download $name. If it is gated (llama, gemma), accept its license on huggingface and set HF_TOKEN"
        done
    fi
fi

cat <<EOF

Done. Activate the venv with:  source $VENV_DIR/bin/activate

The GYAFC formality data can't be downloaded automatically: request access at
https://github.com/raosudha89/GYAFC-corpus and replace the dummy folders in train_data/GYAFC_Corpus/.
EOF
