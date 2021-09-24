#!/usr/bin/env bash

set -euo pipefail

# Download sentence model weights
if [[ ! -d wikidata_linker/sent_model/ ]]; then
    wget https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/v0.2/stsb-roberta-base.zip
    unzip stsb-roberta-base.zip -d stsb-roberta-base
    mv stsb-roberta-base wikidata_linker/sent_model/
    rm stsb-roberta-base.zip
fi

# Create kgtk cache
if [[ ! -d wikidata_linker/kgtk_cache/ ]]; then
    mkdir wikidata_linker/kgtk_cache
fi

# Download requirements into Conda or Venv environment
pip install -r requirements.txt