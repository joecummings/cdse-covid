"""Takes the claim data and uses AMR graphs to extract claimers and x-variables.

You will need to run this in your transition-amr virtual environment.
"""
import argparse
import logging
from pathlib import Path

import spacy
from spacy.language import Language
import torch

from cdse_covid.claim_detection.run_claim_detection import ClaimDataset
from cdse_covid.pegasus_pipeline.run_amr_parsing_all import tokenize_sentence
from cdse_covid.semantic_extraction.models.amr import AMRModel
from cdse_covid.semantic_extraction.run_wikidata_linking import get_best_qnode_for_mention_text
from cdse_covid.semantic_extraction.utils.amr_extraction_utils import (
    identify_x_variable,
    identify_x_variable_covid,
)
from cdse_covid.semantic_extraction.utils.claimer_utils import identify_claimer
from wikidata_linker.get_claim_semantics import get_claim_semantics
from wikidata_linker.linker import WikidataLinkingClassifier
from wikidata_linker.wikidata_linking import CPU

COVID_DOMAIN = "covid"


def main(
    input_dir: Path,
    output: Path,
    *,
    max_tokens: int,
    spacy_model: Language,
    parser_path: Path,
    state_dict: Path,
    domain: str,
    device: str = CPU,
) -> None:
    """Entrypoint to AMR parsing script."""
    amr_parser = AMRModel.from_folder(parser_path)

    linking_model = WikidataLinkingClassifier()
    model_ckpt = torch.load(state_dict, map_location=torch.device(device))
    linking_model.load_state_dict(model_ckpt, strict=False)
    linking_model.to(device)

    spacy_tokenizer = spacy_model.tokenizer

    claim_ds = ClaimDataset.load_from_key_value_store(input_dir)

    for claim in claim_ds.claims:
        logging.info("Processing claim %s", claim.claim_id)
        tokenized_sentence = tokenize_sentence(claim.claim_sentence, spacy_tokenizer, max_tokens)
        sentence_amr = amr_parser.amr_parse_sentences([tokenized_sentence])
        tokenized_claim = tokenize_sentence(claim.claim_text, spacy_tokenizer, max_tokens)
        possible_claimer = identify_claimer(
            claim, tokenized_claim, sentence_amr.graph, sentence_amr.alignments, spacy_model
        )
        if possible_claimer:
            # Add claimer data to Claim
            claim.claimer = possible_claimer
            best_qnode = get_best_qnode_for_mention_text(
                possible_claimer,
                claim,
                sentence_amr.graph,
                sentence_amr.alignments,
                spacy_model,
                linking_model,
                device,
            )
            if best_qnode:
                claim.claimer_type_qnode = best_qnode

        claim_amr = amr_parser.amr_parse_sentences([tokenized_claim])

        if domain == COVID_DOMAIN:
            possible_x_variable = identify_x_variable_covid(
                claim_amr.graph, claim_amr.alignments, claim, spacy_tokenizer
            )
        else:
            claim_ents = {ent.text: ent.label_ for ent in spacy_model(claim.claim_text).ents}
            claim_pos = {token.text: token.pos_ for token in spacy_model(claim.claim_text).doc}
            possible_x_variable = identify_x_variable(
                claim_amr.graph, claim_amr.alignments, claim, claim_ents, claim_pos, spacy_tokenizer
            )
        if possible_x_variable:
            claim.x_variable = possible_x_variable

        # Get claim semantics from AMR data
        semantics = get_claim_semantics(
            claim_amr.graph, claim_amr.alignments, claim, spacy_model, linking_model, device
        )
        claim.claim_semantics = semantics

        claim.add_theory("amr", sentence_amr.graph)
        claim.add_theory("alignments", sentence_amr.alignments)

    claim_ds.save_to_key_value_store(output)

    logging.info("AMR output saved to %s", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Input docs", type=Path)
    parser.add_argument("--output", help="AMR output dir", type=Path)
    parser.add_argument("--amr-parser-model", type=Path)
    parser.add_argument("--state-dict", type=Path)
    parser.add_argument(
        "--max-tokens", help="Max tokens allowed in a sentence to be parsed", type=int, default=50
    )
    parser.add_argument("--domain", help="`covid` or `general`", type=str, default="general")
    parser.add_argument("--device", help="cpu or cuda", type=str, default=CPU)

    args = parser.parse_args()

    model = spacy.load("en_core_web_md")

    main(
        args.input,
        args.output,
        max_tokens=args.max_tokens,
        spacy_model=model,
        parser_path=args.amr_parser_model,
        state_dict=args.state_dict,
        domain=args.domain,
        device=args.device,
    )
