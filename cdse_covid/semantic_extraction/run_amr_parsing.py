"""
Takes the claim data and uses AMR graphs to extract claimers and x-variables.

You will need to run this in your transition-amr virtual environment.
"""
import argparse
import logging
from pathlib import Path

import spacy
from spacy.language import Language

from cdse_covid.claim_detection.run_claim_detection import ClaimDataset
from cdse_covid.pegasus_pipeline.run_amr_parsing_all import tokenize_sentence
from cdse_covid.semantic_extraction.models.amr import AMRModel
from cdse_covid.semantic_extraction.utils.amr_extraction_utils import (
    identify_x_variable,
    identify_x_variable_covid,
)
from cdse_covid.semantic_extraction.utils.claimer_utils import identify_claimer

COVID_DOMAIN = "covid"


def main(
    input_dir: Path,
    output: Path,
    *,
    max_tokens: int,
    spacy_model: Language,
    parser_path: Path,
    domain: str,
) -> None:

    amr_parser = AMRModel.from_folder(parser_path)

    claim_ds = ClaimDataset.load_from_dir(input_dir)

    for claim in claim_ds.claims:
        tokenized_sentence = tokenize_sentence(
            claim.claim_sentence, spacy_model.tokenizer, max_tokens
        )
        sentence_amr = amr_parser.amr_parse_sentences([tokenized_sentence])
        tokenized_claim = tokenize_sentence(claim.claim_text, spacy_model.tokenizer, max_tokens)
        possible_claimer = identify_claimer(
            tokenized_claim, sentence_amr.graph, sentence_amr.alignments
        )
        if possible_claimer:
            claim.claimer = possible_claimer

        claim_amr = amr_parser.amr_parse_sentences([tokenized_claim])

        if domain == COVID_DOMAIN:
            possible_x_variable = identify_x_variable_covid(
                claim_amr.graph, claim_amr.alignments, claim.claim_template
            )
        else:
            claim_ents = {ent.text: ent.label_ for ent in spacy_model(claim.claim_text).ents}
            claim_pos = {token.text: token.pos_ for token in spacy_model(claim.claim_text).doc}
            possible_x_variable = identify_x_variable(
                claim_amr.graph, claim_amr.alignments, claim_ents, claim_pos
            )
        if possible_x_variable:
            claim.x_variable = possible_x_variable

    claim_ds.save_to_dir(output)

    logging.info("AMR output saved to %s", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Input docs", type=Path)
    parser.add_argument("--output", help="AMR output dir", type=Path)
    parser.add_argument("--amr-parser-model", type=Path)
    parser.add_argument(
        "--max-tokens", help="Max tokens allowed in a sentence to be parsed", type=int, default=50
    )
    parser.add_argument("--domain", help="`covid` or `general`", type=str, default="general")

    args = parser.parse_args()

    model = spacy.load("en_core_web_sm")

    main(
        args.input,
        args.output,
        max_tokens=args.max_tokens,
        spacy_model=model,
        parser_path=args.amr_parser_model,
        domain=args.domain,
    )
