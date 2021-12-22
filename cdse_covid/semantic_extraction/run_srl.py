"""Run SRL over claims to get X Variable."""
import argparse
import logging
from pathlib import Path

import spacy
from spacy.language import Language

from cdse_covid.claim_detection.run_claim_detection import ClaimDataset
from cdse_covid.semantic_extraction.models.srl import SRLModel
from cdse_covid.semantic_extraction.utils.amr_extraction_utils import create_x_variable


def reformat_x_variable_in_claim_template(claim_template: str, reference_word: str = "this") -> str:
    """Replaces 'X' in claim template with reference word.

    TODO: Investigate how SRL deals with Person-X, Animal-X, etc.
    """
    template = []
    for token in claim_template.split():
        if token == "X":
            template.append(reference_word)
        else:
            template.append(token)
    return " ".join(template)


def main(inputs: Path, output: Path, *, spacy_model: Language) -> None:
    """Entrypoint to srl script."""
    srl_model = SRLModel.from_hub("structured-prediction-srl", spacy_model)
    claim_ds = ClaimDataset.load_from_key_value_store(inputs)

    for claim in claim_ds.claims:
        srl_out = srl_model.predict(claim.claim_text)

        # Find X variable if it wasn't found in the AMR step
        if claim.x_variable is None and claim.claim_template:
            claim_template = reformat_x_variable_in_claim_template(claim.claim_template)
            srl_claim_template = srl_model.predict(claim_template)
            arg_label_for_x_variable = [
                k for k, v in srl_claim_template.args.items() if v == "this"
            ]
            if arg_label_for_x_variable:
                label = arg_label_for_x_variable[0]  # Should only be one
                x_variable = srl_out.args.get(label)
                if x_variable:
                    claim.x_variable = create_x_variable(x_variable, claim)

        claim.add_theory("srl", srl_out)
    claim_ds.save_to_key_value_store(output)

    logging.info("Finished saving SRL labels to %s.", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Input docs", type=Path)
    parser.add_argument("--output", help="Out file", type=Path)
    parser.add_argument("--spacy-model", type=Path)
    args = parser.parse_args()

    model = spacy.load("en_core_web_md")

    main(args.input, args.output, spacy_model=model)
