"""Run WikiData over claim semantics."""
import argparse
import logging
from pathlib import Path
import re
from typing import Any, List, Optional

from amr_utils.alignments import AMR_Alignment
from amr_utils.amr import AMR
import spacy
from spacy.language import Language
import torch

from cdse_covid.claim_detection.claim import Claim
from cdse_covid.claim_detection.run_claim_detection import ClaimDataset
from cdse_covid.semantic_extraction.mentions import Mention, WikidataQnode
from cdse_covid.semantic_extraction.utils.amr_extraction_utils import PROPBANK_PATTERN
from wikidata_linker.get_claim_semantics import STOP_WORDS, determine_best_qnode, load_tables
from wikidata_linker.linker import WikidataLinkingClassifier
from wikidata_linker.wikidata_linking import (
    CPU,
    REFVAR,
    VERB,
    disambiguate_refvar_kgtk,
    disambiguate_verb_kgtk,
)


def find_links(
    span: str,
    query: str,
    query_type: str,
    linking_model: WikidataLinkingClassifier,
    device: str = CPU,
) -> Any:
    """Find WikiData links for a set of tokens.

    Assumes the query is a refvar by default.
    """
    if query_type == VERB:
        return disambiguate_verb_kgtk(query, linking_model, k=1, device=device)
    else:
        return disambiguate_refvar_kgtk(query, linking_model, span, k=1, device=device)


def get_best_qnode_for_mention_text(
    mention: Mention,
    claim: Claim,
    amr: AMR,
    alignments: List[AMR_Alignment],
    spacy_model: Language,
    linking_model: WikidataLinkingClassifier,
    device: str,
) -> Optional[WikidataQnode]:
    """Return the best WikidataQnode for a string within the claim sentence.

    First, if the string comes from a propbank frame, try a DWD lookup.
    Otherwise, run KGTK.
    """
    mention_text = mention.text
    if not mention_text:
        return None
    # Make both tables
    pbs_to_qnodes_master, pbs_to_qnodes_overlay = load_tables()

    # Find the label associated with the last token of the variable text
    # (any tokens before it are likely modifiers)
    variable_node_label = None
    claim_variable_tokens = [
        mention_token
        for mention_token in mention_text.split(" ")
        if mention_token not in STOP_WORDS
    ]
    claim_variable_tokens.reverse()

    # Locate the AMR node label associated with the mention text
    for node in amr.nodes:
        token_list_for_node = amr.get_tokens_from_node(node, alignments)
        # Try to match the last token since it is most likely to point to the correct node
        if claim_variable_tokens[0] in token_list_for_node:
            variable_node_label = amr.nodes[node].strip('"')

    if not variable_node_label:
        logging.warning(
            "DWD lookup: could not find AMR node corresponding with XVariable/Claimer '%s'",
            mention_text,
        )

    elif re.match(PROPBANK_PATTERN, variable_node_label):
        best_qnode = determine_best_qnode(
            variable_node_label,
            pbs_to_qnodes_overlay,
            pbs_to_qnodes_master,
            amr,
            spacy_model,
            linking_model,
            check_mappings_only=True,
        )
        if best_qnode:
            return WikidataQnode(
                text=best_qnode.get("name"),
                mention_id=mention.mention_id,
                doc_id=claim.doc_id,
                span=mention.span,
                description=best_qnode.get("definition"),
                from_query=best_qnode.get("pb"),
                qnode_id=best_qnode.get("qnode"),
            )
    # If no Qnode was found, try KGTK
    query_list: List[Optional[str]] = [mention.text, variable_node_label, *claim_variable_tokens]
    for query in list(filter(None, query_list)):
        claim_variable_links = find_links(
            claim.claim_sentence, query, REFVAR, linking_model, device
        )
        claim_event_links = find_links(claim.claim_sentence, query, VERB, linking_model, device)
        # Combine the results
        all_claim_links = claim_variable_links
        all_claim_links["options"].extend(claim_event_links["options"])
        all_claim_links["other_options"].extend(claim_event_links["other_options"])
        all_claim_links["all_options"].extend(claim_event_links["all_options"])
        top_link = create_wikidata_qnodes(all_claim_links, mention, claim)
        if top_link:
            return top_link
    return None


def main(
    claim_input: Path,
    srl_input: Path,
    amr_input: Path,
    state_dict: Path,
    output: Path,
    spacy_model: Language,
    device: str = CPU,
) -> None:
    """Entry point to linking script."""
    ds1 = ClaimDataset.load_from_dir(claim_input)
    ds2 = ClaimDataset.load_from_dir(srl_input)
    ds3 = ClaimDataset.load_from_dir(amr_input)
    claim_dataset = ClaimDataset.from_multiple_claims_ds(ds1, ds2, ds3)

    linking_model = WikidataLinkingClassifier()
    model_ckpt = torch.load(state_dict, map_location=torch.device(device))
    linking_model.load_state_dict(model_ckpt, strict=False)
    linking_model.to(device)

    for claim in claim_dataset:
        claim_amr = claim.get_theory("amr")
        claim_alignments = claim.get_theory("alignments")
        if claim_amr and claim_alignments:
            if claim.x_variable:
                best_qnode = get_best_qnode_for_mention_text(
                    claim.x_variable,
                    claim,
                    claim_amr,
                    claim_alignments,
                    spacy_model,
                    linking_model,
                    device,
                )
                if best_qnode:
                    claim.x_variable_identity_qnode = best_qnode
        else:
            logging.warning(
                "Could not load AMR or alignments for claim sentence '%s'",
                claim.claim_sentence,
            )

    claim_dataset.save_to_dir(output)

    logging.info("Saved claims with Wikidata to %s", output)


def create_wikidata_qnodes(link: Any, mention: Mention, claim: Claim) -> Optional[WikidataQnode]:
    """Create WikiData Qnodes from links."""
    if len(link["options"]) < 1:
        all_options = link["all_options"]
        if len(all_options) < 1:
            logging.warning("No WikiData links found for '%s'.", link["query"])
            return None
        else:
            first_option = all_options[0]
            text = first_option["label"][0] if first_option["label"] else None
            qnode = first_option["qnode"][0] if first_option["qnode"] else None
            description = first_option["description"][0] if first_option["description"] else None
    else:
        text = link["options"][0]["rawName"]
        qnode = link["options"][0]["qnode"]
        description = link["options"][0]["definition"]

    return WikidataQnode(
        text=text,
        mention_id=mention.mention_id,
        doc_id=claim.doc_id,
        span=mention.span,
        qnode_id=qnode,
        description=description,
        from_query=link["query"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-input", type=Path)
    parser.add_argument("--srl-input", type=Path)
    parser.add_argument("--amr-input", type=Path)
    parser.add_argument("--state-dict", type=Path, help="Path to `wikidata_classifier.state_dict`")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    model = spacy.load("en_core_web_md")

    main(
        args.claim_input,
        args.srl_input,
        args.amr_input,
        args.state_dict,
        args.output,
        model,
        args.device,
    )
