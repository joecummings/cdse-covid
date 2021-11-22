"""Run WikiData over claim semantics."""
import argparse
import logging
import re
from pathlib import Path
from typing import Any, Optional

from cdse_covid.claim_detection.claim import Claim
from cdse_covid.claim_detection.run_claim_detection import ClaimDataset
from cdse_covid.semantic_extraction.mentions import WikidataQnode
from cdse_covid.semantic_extraction.utils.amr_extraction_utils import PROPBANK_PATTERN
from wikidata_linker.disambiguate_with_amr import load_tables, determine_best_qnode
from wikidata_linker.wikidata_linking import disambiguate_kgtk


def find_links(span: str, query: str) -> Any:
    """Find WikiData links for a set of tokens."""
    return disambiguate_kgtk(span, query, k=1)


def get_best_qnode_for_xvariable_or_claimer(
        claim_variable: str, claim: Claim
) -> Optional[WikidataQnode]:
    """Return the best WikidataQnode for either an XVariable or a Claimer.

    First, if the variable comes from a propbank frame, try a DWD lookup.
    Otherwise, run KGTK.
    """
    # Make both tables
    pbs_to_qnodes_master, pbs_to_qnodes_overlay = load_tables()

    amr = claim.get_theory("amr")
    alignments = claim.get_theory("alignments")

    # Find the label associated with the last token of the variable text
    # (any tokens before it are likely modifiers)
    variable_node_label = None
    claim_variable_last_token = claim_variable.rsplit(" ")[-1]
    for node in amr.nodes:
        token_list_for_node = amr.get_tokens_from_node(node, alignments)
        if claim_variable_last_token in token_list_for_node:
            variable_node_label = amr.nodes[node]

    if not variable_node_label:
        logging.warning(
            "DWD lookup: could not find AMR node corresponding with XVariable/Claimer '%s'",
            claim_variable
        )

    elif re.match(PROPBANK_PATTERN, variable_node_label):
        best_qnode = determine_best_qnode(
            [variable_node_label],
            pbs_to_qnodes_overlay,
            pbs_to_qnodes_master,
            amr,
            check_mappings_only=True
        )
        if best_qnode:
            return WikidataQnode(
                text=best_qnode.get("name"),
                doc_id=claim.doc_id,
                description=best_qnode.get("definition"),
                from_query=best_qnode.get("pb"),
                qnode_id=best_qnode.get("qnode"),
            )
    # If no Qnode was found, try KGTK
    claim_variable_links = find_links(claim.claim_sentence, claim_variable)
    top_link = create_wikidata_qnodes(claim_variable_links, claim)
    if top_link:
        return top_link
    return None


def main(claim_input: Path, srl_input: Path, amr_input: Path, output: Path) -> None:
    """Entry point to linking script."""
    ds1 = ClaimDataset.load_from_dir(claim_input)
    ds2 = ClaimDataset.load_from_dir(srl_input)
    ds3 = ClaimDataset.load_from_dir(amr_input)
    claim_dataset = ClaimDataset.from_multiple_claims_ds(ds1, ds2, ds3)

    for claim in claim_dataset:
        if claim.claimer:
            best_qnode = get_best_qnode_for_xvariable_or_claimer(claim.claimer.text, claim)
            if best_qnode:
                claim.claimer_qnode = best_qnode
        if claim.x_variable:
            best_qnode = get_best_qnode_for_xvariable_or_claimer(claim.x_variable.text, claim)
            if best_qnode:
                claim.x_variable_qnode = best_qnode

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
