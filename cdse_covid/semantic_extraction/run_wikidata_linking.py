import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Optional, Sequence

from cdse_covid.claim_detection.run_claim_detection import ClaimDataset
from wikidata_linker.wikidata_linking import disambiguate_kgtk

from cdse_covid.semantic_extraction.claimer import identify_claimer

@dataclass
class WikidataQnode:
    qnode_id: str
    label: str
    description: Optional[str] = None
    score: Optional[float] = None
    from_query: Optional[str] = None


def _find_links(span, tokens: Sequence[str]):
    """Find WikiData links for a set of tokens."""
    return (disambiguate_kgtk(span, token, thresh=0.25, k=1) for token in tokens)


def main(claim_input, srl_input, amr_input, output):
    ds1 = ClaimDataset.load_from_dir(claim_input)
    ds2 = ClaimDataset.load_from_dir(srl_input)
    ds3 = ClaimDataset.load_from_dir(amr_input)
    claim_dataset = ClaimDataset.from_multiple_claims_ds(ds1, ds2, ds3)

    for claim in claim_dataset:
        wikidata = []
        possible_claimers = identify_claimer(claim.get_theory("amr").graph)
        if possible_claimers:
            claimer_links = _find_links(claim.text, possible_claimers)
            top_links = [
                WikidataQnode(
                    link["all_options"][0]["qnode"],
                    link["all_options"][0]["label"][0],
                    link["all_options"][0]["description"][0],
                    link["all_options"][0]["score"],
                    link["query"]
                ) for link in claimer_links]
            wikidata.extend(top_links)
        for _, sr_label in claim.get_theory("srl").labels.items():
            srl_links = _find_links(claim.text, sr_label.split())
            top_links = [
                WikidataQnode(
                    link["all_options"][0]["qnode"],
                    link["all_options"][0]["label"][0],
                    link["all_options"][0]["description"][0],
                    link["all_options"][0]["score"],
                    link["query"]
                ) for link in srl_links]
            wikidata.extend(top_links)

        claim.add_theory("wikidata", wikidata)

    claim_dataset.save_to_dir(output)

    logging.info("Saved claims with Wikidata to %s", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-input", type=Path)
    parser.add_argument("--srl-input", type=Path)
    parser.add_argument("--amr-input", type=Path)
    parser.add_argument("--output", type=Path)

    args = parser.parse_args()

    from cdse_covid.semantic_extraction.run_wikidata_linking import WikidataQnode

    main(
        args.claim_input,
        args.srl_input,
        args.amr_input,
        args.output,
    )
