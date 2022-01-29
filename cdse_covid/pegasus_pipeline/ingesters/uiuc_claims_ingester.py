"""Ingest claims from UIUC's internal format."""
import argparse
import json
import logging
from pathlib import Path

import spacy
from spacy import Language

from cdse_covid.claim_detection.claim import TOKEN_OFFSET_THEORY, Claim, create_id
from cdse_covid.claim_detection.run_claim_detection import ClaimDataset


def main(claims_file: Path, output: Path, spacy_model: Language) -> None:
    """Entrypoint to UIUC claims ingestion."""
    with open(claims_file, "r", encoding="utf-8") as handle:
        all_claims = json.load(handle)

    isi_claim_dataset = ClaimDataset()

    valid_claims = 0
    invalid_claims = 0
    for doc_id, claims in all_claims.items():
        for claim in claims:
            if not claim["claim_span_text"]:
                invalid_claims += 1
                continue

            # Get offsets from tokens in the claim's sentence
            claim_sentence_tokens_to_offsets = {}
            sentence_start = claim["start_char"]
            idx = sentence_start
            local_idx = 0
            tokenized_sentence = spacy_model.tokenizer(claim["sentence"])
            sentence = [token.text for token in tokenized_sentence]
            for token_idx, sentence_token in enumerate(sentence):
                claim_sentence_tokens_to_offsets[sentence_token] = (idx, idx + len(sentence_token))
                local_idx += len(sentence_token)
                if token_idx + 1 < len(sentence):
                    idx = sentence_start + claim["sentence"].find(
                        sentence[token_idx + 1], local_idx
                    )  # Should account for whitespace

            new_claim = Claim(
                claim_id=create_id(),
                doc_id=doc_id,
                claim_text=claim["claim_span_text"],
                claim_span=(
                    sentence_start + claim["claim_span_start"],
                    sentence_start + claim["claim_span_end"],
                ),
                claim_sentence=claim["sentence"],
            )

            new_claim.add_theory("origins", claim["segment_id"])
            new_claim.add_theory(TOKEN_OFFSET_THEORY, claim_sentence_tokens_to_offsets)

            isi_claim_dataset.add_claim(new_claim)
            valid_claims += 1

    isi_claim_dataset.save_to_key_value_store(output)
    logging.info(
        "Done ingesting claims from UIUC output - %s per valid claims",
        valid_claims / (valid_claims + invalid_claims),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims-file", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    model = spacy.load("en_core_web_md")
    main(args.claims_file, args.output, model)
