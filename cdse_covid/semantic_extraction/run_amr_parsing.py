"""Takes the claim data and uses AMR graphs to extract claimers and x-variables.

You will need to run this in your transition-amr virtual environment.
"""
import argparse
import logging
from pathlib import Path

from nltk.stem import WordNetLemmatizer
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
from wikidata_linker.qnode_mapping_utils import load_tables
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
    max_batch_size: int = 8,
    device: str = CPU,
) -> None:
    """Entrypoint to AMR parsing script."""
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    amr_parser = AMRModel.from_folder(parser_path)

    linking_model = WikidataLinkingClassifier()
    model_ckpt = torch.load(state_dict, map_location=torch.device(device))
    linking_model.load_state_dict(model_ckpt, strict=False)
    linking_model.to(device)

    spacy_tokenizer = spacy_model.tokenizer
    wordnet_lemmatizer = WordNetLemmatizer()

    qnode_mappings = load_tables()

    claim_ds = ClaimDataset.load_from_key_value_store(input_dir)

    # First, collect the sentences and claim text
    tokenized_sentences = []
    claims_to_sentences = {}
    tokenized_claims = []
    claims_to_claim_text = {}
    for claim in claim_ds.claims:
        # Eval hack: filter out image descriptions
        if "data-image-title" in claim.claim_sentence:
            continue
        tokenized_sentence = tokenize_sentence(claim.claim_sentence, spacy_tokenizer, max_tokens)
        if len(tokenized_sentence) <= 1:
            continue
        tokenized_claim = tokenize_sentence(claim.claim_text, spacy_tokenizer, max_tokens)
        if len(tokenized_claim) <= 1:
            continue
        tokenized_sentences.append(tokenized_sentence)
        claims_to_sentences[claim.claim_id] = " ".join(tokenized_sentence)
        tokenized_claims.append(tokenized_claim)
        claims_to_claim_text[claim.claim_id] = " ".join(tokenized_claim)

    # Parse the sentences/claims
    logger.info("Beginning AMR parsing...")
    all_sentence_amr_data = amr_parser.amr_parse_sentences(tokenized_sentences)
    all_claim_amr_data = amr_parser.amr_parse_sentences(tokenized_claims)
    logger.info("AMR parsing complete.")

    for claim in claim_ds.claims:
        # Find the AMR data for each claim
        tokenized_claim_string = claims_to_claim_text.get(claim.claim_id)
        if not tokenized_claim_string:
            continue
        sentence_amr_data = all_sentence_amr_data.get(claims_to_sentences[claim.claim_id])
        if not sentence_amr_data:
            continue
        claim_amr_data = all_claim_amr_data.get(tokenized_claim_string)
        if not claim_amr_data:
            continue

        # Extract the claim semantics from the AMR data
        logger.info("Processing claim %s", claim.claim_id)
        possible_claimer = identify_claimer(
            claim,
            tokenized_claim_string.split(" "),
            sentence_amr_data.graph,
            sentence_amr_data.alignments,
            spacy_model,
        )
        if possible_claimer:
            # Add claimer data to Claim
            claim.claimer = possible_claimer
            best_qnode = get_best_qnode_for_mention_text(
                possible_claimer,
                claim,
                sentence_amr_data.graph,
                sentence_amr_data.alignments,
                spacy_model,
                linking_model,
                wordnet_lemmatizer,
                qnode_mappings,
                max_batch_size,
                device,
            )
            if best_qnode:
                claim.claimer_type_qnode = best_qnode

        if domain == COVID_DOMAIN:
            possible_x_variable = identify_x_variable_covid(
                claim_amr_data.graph, claim_amr_data.alignments, claim, spacy_tokenizer
            )
        else:
            claim_ents = {ent.text: ent.label_ for ent in spacy_model(claim.claim_text).ents}
            claim_pos = {token.text: token.pos_ for token in spacy_model(claim.claim_text).doc}
            possible_x_variable = identify_x_variable(
                claim_amr_data.graph,
                claim_amr_data.alignments,
                claim,
                claim_ents,
                claim_pos,
                spacy_tokenizer,
            )
        if possible_x_variable:
            claim.x_variable = possible_x_variable

        # Get claim semantics from AMR data
        semantics = get_claim_semantics(
            sentence_amr_data.graph,
            sentence_amr_data.alignments,
            claim,
            spacy_model,
            linking_model,
            wordnet_lemmatizer,
            qnode_mappings,
            max_batch_size,
            device,
        )
        claim.claim_semantics = semantics

        claim.add_theory("amr", sentence_amr_data.graph)
        claim.add_theory("alignments", sentence_amr_data.alignments)

    claim_ds.save_to_key_value_store(output)

    logger.info("AMR output saved to %s", output)


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
    parser.add_argument(
        "--max-batch-size", help="Max batch size; 8 is recommended", type=int, default="8"
    )
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
        max_batch_size=args.max_batch_size,
        device=args.device,
    )
