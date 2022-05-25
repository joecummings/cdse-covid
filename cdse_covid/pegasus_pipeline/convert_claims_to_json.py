"""Merge all annotated claims into one JSON file."""
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
    """Convert claim to JSON format."""
    return Claim.to_json(claim)


def main(input_dir: Path, output: Path) -> None:
    """Entrypoint to merge script."""
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    claims = ClaimDataset.load_from_key_value_store(input_dir)
    structured_claims = [structure_claim(claim) for claim in claims]
    structured_claims.sort(key=lambda x: (x["doc_id"], x["claim_id"]))  # type:ignore
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w+", encoding="utf-8") as handle:
        json.dump(structured_claims, handle)

    logging.info("Saved merged claims to %s", output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    main(args.input, args.output)
