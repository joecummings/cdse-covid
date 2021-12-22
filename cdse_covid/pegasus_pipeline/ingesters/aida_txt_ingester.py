"""Ingest AIDA English TXT files and run SpaCy tokenization."""
import argparse
import logging
from pathlib import Path

from spacy import load
from spacy.language import Language


def load_aida_txt_docs(docs_to_load: Path, output_dir: Path, *, spacy_model: Language) -> None:
    """Load AIDA docs in rsd.txt format and encode with SpaCy."""
    for txt_file in docs_to_load.glob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as handle:
            doc_text = handle.read()
            doc_id = txt_file.stem.strip(".rsd")
            doc = spacy_model(doc_text)
            doc.to_disk(f"{output_dir / doc_id}.spacy")

    logging.info("Ingested all AIDA text documents!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--spacy-model", type=Path)
    args = parser.parse_args()

    model = load(args.spacy_model)

    load_aida_txt_docs(args.corpus, args.output, spacy_model=model)
