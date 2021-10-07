#!/usr/bin/env bash

set -euo pipefail

ISI_USERNAME=${1:-""}

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

# Transition AMR Parser installation

source ~/miniconda3/etc/profile.d/conda.sh
set +u  # hack for conda issue
conda deactivate
if ! conda env list | grep -q 'transition-amr-parser'; then
  echo "Creating a new conda environment for the AMR parser..."
  echo y | conda create -n transition-amr-parser python=3.7
fi
conda activate transition-amr-parser
set -u  # /hack

echo "Installing transition-amr-parser..."
cd ..
if [[ ! -d transition-amr-parser/ ]]; then
  git clone https://github.com/IBM/transition-amr-parser.git
fi
cd transition-amr-parser || { echo "Could not navigate to transition-amr-parser"; exit 1; }
touch set_environment.sh
python -m pip install -e .
# fairseq loading fix
sed -i ".bak" "s/pytorch\/fairseq/\pytorch\/fairseq\:main/" transition_amr_parser/parse.py
echo "Running installation test..."
bash tests/correctly_installed.sh
if ! bash tests/correctly_installed.sh | grep -q 'correctly installed'; then
  echo "AMR parser not correctly installed -- check to make sure that each requirement has been installed properly"
  exit 1
fi
echo "Parser installed (1/4)"

echo "Installing JAMR aligner..."
cd preprocess || { echo "Could not navigate to $(pwd)/preprocess"; exit 1; }
rm -Rf jamr
git clone https://github.com/jflanigan/jamr.git
if [[ ! -e ~/.sbt/repositories ]]; then
  printf "[repositories]\n\tmaven-central: https://repo1.maven.org/maven2/" > ~/.sbt/repositories
fi
cd jamr || { echo "Could not navigate to $(pwd)/jamr"; exit 1; }
git checkout Semeval-2016

# Remove troublesome lines if they still exist
BUILD_FILE="build.sbt"
PLUGIN_FILE="project/plugins.sbt"
grep -v "import AssemblyKeys._" $BUILD_FILE > tmpfile && mv tmpfile $BUILD_FILE
grep -v "assemblySettings" $BUILD_FILE > tmpfile && mv tmpfile $BUILD_FILE
grep -v "sbt-idea" $PLUGIN_FILE > tmpfile && mv tmpfile $PLUGIN_FILE

# Update package versions
echo "Updating package versions for JAMR..."
sed -i ".bak" "s/\"scala-arm\" % \"[0-9]*\.[0-9]*\"/\"scala-arm\" % \"2\.0\"/" $BUILD_FILE
sed -i ".bak" "s/\"sbt-assembly\" % \"[0-9]*\.[0-9]*\.[0-9]*\"/\"sbt-assembly\" % \"0\.14\.6\"/" $PLUGIN_FILE
sed -i ".bak" "s/\"sbteclipse-plugin\" % \"[0-9]*\.[0-9]*\.[0-9]*\"/\"sbteclipse-plugin\" % \"5\.2\.4\"/" $PLUGIN_FILE
echo "sbt.version=1.2.0" > project/build.properties

./setup || { echo "JAMR setup failed; you may need to further update the config files"; exit 1; }
. scripts/config.sh
cd ..

echo "JAMR installed (2/4)"

echo "Installing Kevin aligner..."
# Install cmake through brew if installed, else use pip
brew install cmake || python -m pip install cmake
if [[ ! -d kevin/ ]]; then
  git clone https://github.com/damghani/AMR_Aligner
  mv AMR_Aligner kevin
  cd kevin
  git clone https://github.com/moses-smt/mgiza.git
  cmake .
  make
  make install
  cd ..
else
  echo "Looks like Kevin is already installed"
fi
cd ..
echo "Kevin installed (3/4)"

MODEL_PATH="/nas/gaia/curated-training/repos/transition-amr-parser/DATA/AMR2.0"
MODEL_PATH+="/models/exp_cofill_o8.3_act-states_RoBERTa-large-top24"
MODEL_PATH+="/_act-pos-grh_vmask1_shiftpos1_ptr-lay6-h1_grh-lay123-h2-allprev_1in1out_cam-layall-h2-abuf"
MODEL_PATH+="/ep120-seed42/{checkpoint_best.pt,config.sh,dict.actions_nopos.txt,dict.en.txt,entity_rules.json,train.rules.json}"

cd DATA || { echo "Could not navigate to $(pwd)/DATA"; exit 1; }
if [[ ! -d AMR2.0/ ]]; then
  echo "Downloading model..."
  if [[ $ISI_USERNAME == "" ]]; then
    cp -r $MODEL_PATH . || { echo "Failed to copy model over"; exit 1; }
  else
    scp -r "$ISI_USERNAME"@minlp-dev-01:"$MODEL_PATH" . || { echo "Failed to download model"; exit 1; }
  fi
else
  echo "Looks like the required model is already present"
fi
echo "Done!"
