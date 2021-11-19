"""Collection of Claimer utils."""
import re
from typing import Any, List, MutableMapping, Optional

from amr_utils.alignments import AMR_Alignment  # pylint: disable=import-error
from amr_utils.amr import AMR  # pylint: disable=import-error
from nltk.corpus import framenet
from nltk.stem import WordNetLemmatizer

from cdse_covid.claim_detection.claim import Claim
from cdse_covid.semantic_extraction.mentions import Claimer
from cdse_covid.semantic_extraction.utils.amr_extraction_utils import (
    create_node_to_token_dict,
    get_full_description,
    get_full_name_value,
)

LEMMATIZER = WordNetLemmatizer()


framenet_concepts = ["Statement", "Reasoning"]
FRAMENET_VERBS = set()
for concept in framenet_concepts:
    frame = framenet.frame_by_name(concept)
    lex_units = frame.get("lexUnit")
    for verb in lex_units.keys():
        word, pos = verb.split(".")
        if pos == "v":
            word = word.replace(" ", "-")  # Account for multiple words
            FRAMENET_VERBS.add(word)


def identify_claimer(
    claim: Claim, claim_tokens: List[str], amr: AMR, alignments: List[AMR_Alignment]
) -> Optional[Claimer]:
    """Identify the claimer of the span.

    Finding claim node:
        1. Try to match the claim tokens to a token in the AMR graph
            then work up to find the 'statement' node
        2. If 1 fails, find the first 'statement' node in the graph and
            select that as the claim node

    Finding claimer:
        Once claim node is found, get the claimer argument of the node.
    """
    if not amr:
        return None

    claim_node = get_claim_node(claim_tokens, amr)
    arg_node = get_argument_node(amr, alignments, claim_node)
    if arg_node:
        return Claimer(
            text=arg_node, doc_id=claim.doc_id, span=claim.get_offsets_for_text(arg_node)
        )
    return None


def get_claim_node(claim_tokens: List[str], amr: AMR) -> Optional[str]:
    """Get the head node of the claim."""
    graph_nodes = amr.nodes

    for token in claim_tokens:
        token = token.lower()  # Make sure the token is lowercased
        for node, label in graph_nodes.items():
            label = re.sub(r"(-\d*)", "", label)  # Remove any PropBank numbering
            # We're hoping that at least one nominal/verbial lemma is found
            if (
                LEMMATIZER.lemmatize(token, pos="n") == label
                or LEMMATIZER.lemmatize(token, pos="v") == label
            ):
                pot_claim_node = get_claim_node_from_token(amr, node)
                if pot_claim_node:
                    return pot_claim_node
    # No token match was found. Gets here 53% of the time.
    return search_for_claim_node(graph_nodes)


def is_desired_framenet_node(node_label: str) -> bool:
    """Determine if the node under investigation represents a statement or reasoning event."""
    pot_verb = node_label.rsplit("-", 1)[0]  # E.g. origin from origin-01
    return pot_verb in FRAMENET_VERBS


def get_claim_node_from_token(amr: AMR, node: str, checked_nodes: Any = None) -> Optional[str]:
    """Fetch the claim node by traveling up from a child node within the claim."""
    if checked_nodes is None:
        checked_nodes = set()
    nodes_to_labels = amr.nodes
    parents = amr.get_parents_for_node(node)
    for parent_node in parents:
        if parent_node in checked_nodes:
            continue
        if is_desired_framenet_node(nodes_to_labels[parent_node]):
            return str(parent_node) or None
        checked_nodes.add(parent_node)
        next_check = get_claim_node_from_token(amr, parent_node, checked_nodes)
        if next_check:
            return next_check
    return None


def search_for_claim_node(graph_nodes: MutableMapping[str, Any]) -> Optional[str]:
    """Rule #2: Try finding the statement node by reading through all nodes and returning the first match."""
    for node, label in graph_nodes.items():
        if is_desired_framenet_node(label):
            return node
    return None


def get_argument_node(
    amr: AMR, alignments: List[AMR_Alignment], claim_node: Optional[str]
) -> Optional[str]:
    """Get all argument (claimer) nodes of the claim node."""
    nodes = amr.nodes
    nodes_to_strings = create_node_to_token_dict(amr, alignments)
    amr_dict = amr.edge_mapping()
    node_args = amr_dict.get(claim_node)
    if node_args:
        claimer_nodes = node_args.get(":ARG0")
        if claimer_nodes:
            claimer_node = claimer_nodes[0]  # only get one
            claimer_label = nodes.get(claimer_node)
            if claimer_label in ["person", "organization"]:
                return get_full_name_value(amr_dict, nodes_to_strings, claimer_node)
            return get_full_description(amr_dict, nodes, nodes_to_strings, claimer_node)
    return None
