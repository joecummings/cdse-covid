"""Run WikiData over claim semantics."""
import argparse
import logging
from pathlib import Path

from cdse_covid.claim_detection.run_claim_detection import ClaimDataset
from wikidata_linker.disambiguate_with_amr import disambiguate_with_amr


def main(claim_input: Path, output: Path) -> None:
    """Entry point to linking script."""
    claim_dataset = ClaimDataset.load_from_dir(claim_input)

    for claim in claim_dataset:
        amr_sentence = claim.get_theory("amr")
        amr_alignments = claim.get_theory("alignments")
        if amr_sentence:
            semantics = disambiguate_with_amr(amr_sentence, amr_alignments, claim)
            claim.claim_semantics = semantics

    claim_dataset.save_to_dir(output)

    logging.info("Saved claims with Wikidata to %s", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-input", type=Path)
    parser.add_argument("--output", type=Path)

    args = parser.parse_args()

    main(
        args.claim_input,
        args.output,
    )
