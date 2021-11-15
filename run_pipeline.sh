#!/usr/bin/env bash

set -e
set -o pipefail

# Load params
PARAMS=$1
. "$PARAMS"

# Setup
source $CONDA_PATH
conda activate "$CDSE_COVID_ENV_NAME"
mkdir -p "$CHECKPOINT_DIR"

# Preprocessing
if [[ ! -e "$CHECKPOINT_DIR"/spacified_ckpt ]]; then
  echo "Starting preprocessing..."
  mkdir -p "$SPACIFIED_OUTPUT"
  python "$PROJECT_DIR"/cdse_covid/pegasus_pipeline/ingesters/aida_txt_ingester.py \
      --corpus "$CORPUS_PATH" \
      --output "$SPACIFIED_OUTPUT" \
      --spacy-model "$SPACY_MODEL_PATH"

  touch "$CHECKPOINT_DIR"/spacified_ckpt
  echo "Spacified files are now in $SPACIFIED_OUTPUT"
fi

# AMR-all
conda activate transition-amr-parser
if [[ ! -e "$CHECKPOINT_DIR"/amr_all_ckpt ]]; then
  echo "Starting AMR parsing over all sentences..."
  mkdir -p "$AMR_All_OUTPUT"
  python "$PROJECT_DIR"/cdse_covid/pegasus_pipeline/run_amr_parsing_all.py \
      --corpus "$CORPUS_PATH" \
      --output "$AMR_All_OUTPUT" \
      --amr-parser-model "$PROJECT_DIR"/../transition-amr-parser \
      --max-tokens "$MAX_TOKENS"

  touch "$CHECKPOINT_DIR"/amr_all_ckpt
  echo "AMR graphs are now in $AMR_All_OUTPUT"
fi

# Claim detection
conda activate $CDSE_COVID_ENV_NAME
if [[ ! -e "$CHECKPOINT_DIR"/claims_ckpt ]]; then
  echo "Starting claims detection..."
  mkdir -p "$CLAIMS_OUTPUT"
  python "$PROJECT_DIR"/cdse_covid/claim_detection/run_claim_detection.py \
      --input "$SPACIFIED_OUTPUT" \
      --patterns "$PATTERNS_FILE" \
      --out "$CLAIMS_OUTPUT" \
      --spacy-model "$SPACY_MODEL_PATH"

  touch "$CHECKPOINT_DIR"/claims_ckpt
  echo "Claims are now in $CLAIMS_OUTPUT"
fi

# AMR-claims
conda activate transition-amr-parser
if [[ ! -e "$CHECKPOINT_DIR"/amr_ckpt ]]; then
  echo "Starting AMR parsing on claims..."
  mkdir -p "$AMR_OUTPUT"
  python "$PROJECT_DIR"/cdse_covid/semantic_extraction/run_amr_parsing.py \
      --input "$CLAIMS_OUTPUT" \
      --output "$AMR_OUTPUT" \
      --amr-parser-model "$PROJECT_DIR"/../transition-amr-parser \
      --max-tokens "$MAX_TOKENS" \
      --domain "$DOMAIN"

  touch "$CHECKPOINT_DIR"/amr_ckpt
  echo "Claim data from AMR is now in $AMR_OUTPUT"
fi

# SRL
conda activate $CDSE_COVID_ENV_NAME
if [[ ! -e "$CHECKPOINT_DIR"/srl_ckpt ]]; then
  echo "Starting SRL..."
  mkdir -p "$SRL_OUTPUT"
  python "$PROJECT_DIR"/cdse_covid/semantic_extraction/run_srl.py \
      --input "$AMR_OUTPUT" \
      --output "$SRL_OUTPUT" \
      --spacy-model "$SPACY_MODEL_PATH"

  touch "$CHECKPOINT_DIR"/srl_ckpt
  echo "Claim data from SRL is now in $SRL_OUTPUT"
fi

# Wikidata linking
if [[ ! -e "$CHECKPOINT_DIR"/wikidata_ckpt ]]; then
  echo "Starting Wikidata linking..."
  mkdir -p "$WIKIDATA_OUTPUT"
  python "$PROJECT_DIR"/cdse_covid/semantic_extraction/run_wikidata_linking.py \
      --claim-input "$CLAIMS_OUTPUT" \
      --amr-input "$AMR_OUTPUT" \
      --srl-input "$SRL_OUTPUT" \
      --output "$WIKIDATA_OUTPUT"

  touch "$CHECKPOINT_DIR"/wikidata_ckpt
  echo "Output from Wikidata linking is now in $WIKIDATA_OUTPUT"
fi

# Postprocessing
echo "Merging output..."
python "$PROJECT_DIR"/cdse_covid/pegasus_pipeline/merge.py \
    --input "$WIKIDATA_OUTPUT" \
    --output "$FINAL_OUTPUT_FILE"
echo "Final output has been saved to $FINAL_OUTPUT_FILE"

# Finished; remove checkpoints
for f in "$CHECKPOINT_DIR"/*; do
  echo "Removing checkpoint file $f"
  rm "$f"
done
