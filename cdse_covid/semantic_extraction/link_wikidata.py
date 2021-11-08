import argparse
import logging
from pathlib import Path
from typing import Sequence

from cdse_covid.claim_detection.run_claim_detection import ClaimDataset
from wikidata_linker.wikidata_linking import disambiguate_kgtk

from cdse_covid.semantic_extraction.entities import WikidataQnode


def find_links(span, tokens: Sequence[str]):
    """Find WikiData links for a set of tokens."""
    return (disambiguate_kgtk(span, token, k=1) for token in tokens)


def create_wikidata_qnodes(links):
    all_qnodes = []
    for link in links:
        if not link["options"]:
            continue
        qnode = WikidataQnode(
            link["options"][0]["qnode"],
            link["options"][0]["rawName"],
            link["options"][0]["definition"],
            link["query"],
        )
        all_qnodes.append(qnode)
    return all_qnodes


def main(inputs, output):
    all_ds = [ClaimDataset.load_from_dir(ds) for ds in inputs]
    claim_dataset = ClaimDataset.from_multiple_claims_ds(*all_ds)

    for claim in claim_dataset:
        if claim.claimer:
            claimer_links = find_links(claim.claim_sentence, [claim.claimer])
            top_link = create_wikidata_qnodes(claimer_links)
            if top_link:
                claim.claimer_qnode = top_link[0]
        if claim.x_variable:
            srl_links = find_links(claim.claim_sentence, [claim.x_variable])
            top_link = create_wikidata_qnodes(srl_links)
            if top_link:
                claim.x_variable_qnode = top_link[0]

    claim_dataset.save_to_dir(output)

    logging.info("Saved claims with Wikidata to %s", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="*")
    parser.add_argument("--output", type=Path)

    args = parser.parse_args()

    main(
        args.input,
        args.output,
    )
