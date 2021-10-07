# Claim Detection & Semantics Extraction (Covid-19)

## Installation

1. Clone repo
2. Create virtual environment
3. Make sure that your current Java environment is **Java 8**.
   1. If the setup fails at the JAMR step, check that Java 8 is configured
      for the newly downloaded `transition-amr-parser` project.
4. Run `bash setup.sh [$isi_username]`
   1. This assumes that your conda installation is within `~/miniconda3`.
   2. If you provide `isi_username`, it will assume that you can access
      the `minlp-dev-01` server and that you are working from a local system.
      In that case, you will be prompted for a password after you see
      `"Downloading model..."`
      If not, it will assume that you are working from a `/nas`-mounted server.

## Usage

1. Create the AMR files
   The files in `TXT_FILES` should consist of sentences separated by line.
   ```
   conda activate transition-amr-parser
   python -m amr_parsing.parse_files \
       --input TXT_FILES --out AMR_FILES
   ```
2. Claim detection:
   ```
   conda activate <cdse-covid-env>
   python -m claim_detection.models --input AMR_FILES --patterns claim_detection/topics.json --out OUT_FILE
   ```
