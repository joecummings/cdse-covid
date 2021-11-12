import argparse
import json
import logging
from pathlib import Path
from typing import Any, List, MutableMapping, Union

from cdse_covid.claim_detection.claim import Claim
from cdse_covid.claim_detection.run_claim_detection import ClaimDataset


def structure_claim(
    claim: Claim,
) -> Union[List[MutableMapping[str, Any]], MutableMapping[str, Any], Any]:
    if claim.theories:
        claim.theories = {}  # Hack for now b/c we don't want to see theories
    return Claim.to_json(claim)


def main(input_dir: Path, output: Path) -> None:
    claims = ClaimDataset.load_from_dir(input_dir)
    structured_claims = [structure_claim(claim) for claim in claims]

    with open(output, "w+", encoding="utf-8") as handle:
        json.dump(structured_claims, handle)

    logging.info("Saved merged claims to %s", output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    main(args.input, args.output)
