MODEL=marian_es_en_fisher
CONDITION_LAMBDA=1.0
RUN_ID=$(date +%Y%m%d-%H%M%S)
LOG_DIR=logs/formality/$RUN_ID
mkdir -p "$LOG_DIR"
tag="${MODEL}_lambda${CONDITION_LAMBDA}"

source .venv-new/bin/activate

python -u evaluate_formality.py --model "$MODEL" \
    --ckpt ckpt/formality/predictor_gyafc_entertainment_music/model.pth.tar \
    --dataset_info ckpt/formality/predictor_gyafc_entertainment_music/dataset_info \
    --in_file formality_data/fisher_test_oracle.es \
    --condition_lambda "$CONDITION_LAMBDA" --stats_file "$LOG_DIR/stats_$tag.json" \
    > "$LOG_DIR/preds_$tag.log" 2> "$LOG_DIR/errors_$tag.log"

python eval_formality_metrics.py --pred "$LOG_DIR/preds_$tag.log" \
    --ref formality_data/test.noid.cleaned_0 formality_data/test.noid.cleaned_1 \
    --ckpt ckpt/formality/test_evaluator_gyafc_family_relationships/model.pth.tar \
    --dataset_info ckpt/formality/test_evaluator_gyafc_family_relationships/dataset_info \
    > "$LOG_DIR/results_$tag.log" 2>> "$LOG_DIR/errors_$tag.log"
