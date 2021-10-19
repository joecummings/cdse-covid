import json
import logging
from pathlib import Path
from typing import Any, Mapping


import argparse
from cdse_covid.claim_detection.run_claim_detection import Claim, ClaimDataset

def structure_claim(claim: Claim) -> Mapping[str, Any]:
    qnodes = claim.get_theory("wikidata")
    return {
        "claim_text": claim.claim_text,
        "claim_sentence": claim.claim_sentence,
        "claim template": claim.claim_template,
        "doc": claim.doc_id,
        "qnodes": [(q.qnode_id, q.label) for q in qnodes],
    }


def main(input_dir: Path, output: Path):
    claims = ClaimDataset.load_from_dir(input_dir)
    structured_claims = [structure_claim(claim) for claim in claims]

    with open(output, "w+") as handle:
        json.dump(structured_claims, handle)

    logging.info("Saved merged claims to %s", output)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    main(args.input, args.output)
