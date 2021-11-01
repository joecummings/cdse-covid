"""
Takes the corpus files and creates AMR graphs for each sentence.

You will need to run this in your transition-amr virtual environment.
"""
import argparse
import logging
from os import chdir, getcwd, makedirs
from pathlib import Path
from typing import List
import uuid
import spacy


from transition_amr_parser.parse import AMRParser  # pylint: disable=import-error

from amr_utils.amr_readers import AMR_Reader, Matedata_Parser

from cdse_covid.claim_detection.run_claim_detection import ClaimDataset
from cdse_covid.semantic_extraction.models import AMRLabel
from cdse_covid.semantic_extraction.claimer_utils import identify_claimer, identify_x_variable


def tokenize_sentence(text, spacy_tokenizer) -> List[str]:
    tokens = spacy_tokenizer(text.strip())
    tokenized_sentence = [token.text for token in tokens]
    return tokenized_sentence


def main(input_dir, output, *, spacy_model, parser_path):

    cdse_path = getcwd()

    # We assume that the checkpoint is in this location within the repo
    in_checkpoint = f"{parser_path}/DATA/AMR2.0/models" \
                    "/exp_cofill_o8.3_act-states_RoBERTa-large-top24" \
                    "/_act-pos-grh_vmask1_shiftpos1_ptr-lay6-h1_grh-lay123-h2-allprev" \
                    "_1in1out_cam-layall-h2-abuf/ep120-seed42/checkpoint_best.pt"

    if not Path(in_checkpoint).exists():
        raise RuntimeError(f"Could not find checkpoint file {in_checkpoint}!")
    if not input_dir.exists():
        raise RuntimeError(f"Input directory {args.input} could not be found!")
    if not Path(output).exists():
        makedirs(output)

    chdir(parser_path)
    amr_parser = AMRParser.from_checkpoint(in_checkpoint)
    chdir(cdse_path)

    claim_ds = ClaimDataset.load_from_dir(input_dir)

    for claim in claim_ds.claims:
        tokenized_sentences = tokenize_sentence(claim.claim_sentence, spacy_model.tokenizer)
        annotations = amr_parser.parse_sentences([tokenized_sentences])
        metadata, graph_metadata = Matedata_Parser().readlines(annotations[0][0])
        amr, alignments = AMR_Reader._parse_amr_from_metadata(metadata["tok"], graph_metadata)
        tokenized_claim = tokenize_sentence(claim.claim_text, spacy_model.tokenizer)
        possible_claimer = identify_claimer(tokenized_claim, amr, alignments)
        if possible_claimer:
            claim.claimer = possible_claimer

        claim_annotations = amr_parser.parse_sentences([tokenized_claim])
        claim_metadata, claim_graph_metadata = Matedata_Parser().readlines(
            claim_annotations[0][0]
        )
        claimr, claim_alignments = AMR_Reader._parse_amr_from_metadata(
            claim_metadata["tok"], claim_graph_metadata
        )
        possible_x_variable = identify_x_variable(
            claimr, claim_alignments, claim.claim_template
        )
        if possible_x_variable:
            claim.x_variable = possible_x_variable

        amr_label = AMRLabel((int(uuid.uuid1())), amr, alignments)
        claim.add_theory("amr", amr_label)

    claim_ds.save_to_dir(output)

    logging.info("AMR output saved to %s", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Input docs", type=Path)
    parser.add_argument("--output", help="AMR output dir", type=Path)
    parser.add_argument("--amr-parser-model", type=Path)

    args = parser.parse_args()

    model = spacy.load("en_core_web_sm")

    main(args.input, args.output, spacy_model=model, parser_path=args.amr_parser_model)
