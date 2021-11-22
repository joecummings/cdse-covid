"""Run WikiData over claim semantics."""
import argparse
import logging
from pathlib import Path
from typing import Any, Optional

from cdse_covid.claim_detection.claim import Claim
from cdse_covid.claim_detection.run_claim_detection import ClaimDataset
from cdse_covid.semantic_extraction.mentions import WikidataQnode
from wikidata_linker.wikidata_linking import disambiguate_kgtk


def find_links(span: str, query: str) -> Any:
    """Find WikiData links for a set of tokens."""
    return disambiguate_kgtk(span, query, k=1)


def main(claim_input: Path, srl_input: Path, amr_input: Path, output: Path) -> None:
    """Entry point to linking script."""
    ds1 = ClaimDataset.load_from_dir(claim_input)
    ds2 = ClaimDataset.load_from_dir(srl_input)
    ds3 = ClaimDataset.load_from_dir(amr_input)
    claim_dataset = ClaimDataset.from_multiple_claims_ds(ds1, ds2, ds3)

    for claim in claim_dataset:
        if claim.claimer:
            claimer_links = find_links(claim.claim_sentence, claim.claimer.text)
            top_link = create_wikidata_qnodes(claimer_links, claim)
            if top_link:
                claim.claimer_qnode = top_link
        if claim.x_variable:
            srl_links = find_links(claim.claim_sentence, claim.x_variable.text)
            top_link = create_wikidata_qnodes(srl_links, claim)
            if top_link:
                claim.x_variable_qnode = top_link

    claim_dataset.save_to_dir(output)

    logging.info("Saved claims with Wikidata to %s", output)


def create_wikidata_qnodes(link: Any, claim: Claim) -> Optional[WikidataQnode]:
    """Create WikiData Qnodes from links."""
    if len(link["options"]) < 1:
        if len(link["all_options"]) < 1:
            logging.warning("No WikiData links found for '%s'.", link["query"])
            return None
        else:
            text = link["all_options"][0]["label"][0]
            qnode = link["all_options"][0]["qnode"]
            description = link["all_options"][0]["description"][0]
    else:
        text = link["options"][0]["rawName"]
        qnode = link["options"][0]["qnode"]
        description = link["options"][0]["definition"]

    return WikidataQnode(
        text=text,
        doc_id=claim.doc_id,
        span=claim.get_offsets_for_text(link["query"]),
        qnode_id=qnode,
        description=description,
        from_query=link["query"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-input", type=Path)
    parser.add_argument("--srl-input", type=Path)
    parser.add_argument("--amr-input", type=Path)
    parser.add_argument("--output", type=Path)

    args = parser.parse_args()

    main(
        args.claim_input,
        args.srl_input,
        args.amr_input,
        args.output,
    )
