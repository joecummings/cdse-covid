from amr_utils.alignments import AMR_Alignment
from amr_utils.amr import AMR
from nltk.corpus import framenet
from typing import List, Optional
from nltk.stem import WordNetLemmatizer
import re

from cdse_covid.semantic_extraction.amr_extraction_utils import get_full_name_value, get_full_description, \
    create_node_to_token_dict, create_amr_dict

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


def identify_claimer(claim_tokens, amr: AMR, alignments: List[AMR_Alignment]) -> str:
    """Identify the claimer of the span.

    Finding claim node:
        1. Try to match the claim tokens to a token in the AMR graph \
            then work up to find the 'statement' node
        2. If 1 fails, find the first 'statement' node in the graph and \
            select that as the claim node

    Finding claimer:
        Once claim node is found, get the claimer argument of the node.
    """
    if not amr:
        return ""

    claim_node = get_claim_node(claim_tokens, amr)
    return get_argument_node(amr, alignments, claim_node)


def get_claim_node(claim_tokens: List[str], amr: AMR) -> Optional[str]:
    """Get the head node of the claim."""
    graph_nodes = amr.nodes
    
    for token in claim_tokens:
        token = token.lower() # Make sure the token is lowercased
        for node, label in graph_nodes.items():
            label = re.sub("(-\d*)", "", label)  # Remove any PropBank numbering
            # We're hoping that at least one nominal/verbial lemma is found
            if (
                LEMMATIZER.lemmatize(token, pos="n") == label
                or LEMMATIZER.lemmatize(token, pos="v") == label
            ):
                pot_claim_node = get_claim_node_from_token(node, graph_nodes, amr.edges, 0)
                if pot_claim_node:
                    return pot_claim_node
    # No token match was found. Gets here 53% of the time.
    return search_for_claim_node(graph_nodes)


def is_desired_framenet_node(node_label: str) -> bool:
    """Determine if the node under investigation represents a statement or reasoning event."""
    verb = node_label.rsplit("-", 1)[0]  # E.g. origin from origin-01
    return verb in FRAMENET_VERBS


def get_claim_node_from_token(node, node_dict, edges, i) -> Optional[str]:
    """Fetch the claim node by traveling up from a child node."""
    for parent_node, _, arg_node in edges:
        if arg_node == node:
            # Check if the parent is a claim node / Should you stop at the first one?
            parent_label = node_dict[parent_node]
            if is_desired_framenet_node(parent_label):
                return parent_node
            if i == len(node_dict):  # Iterated through all nodes without success
                break
            return get_claim_node_from_token(parent_node, node_dict, edges, i + 1)


def search_for_claim_node(graph_nodes) -> Optional[str]:
    """Rule #2: try finding the statement node by reading through all nodes
    and returning the first match.
    """
    for node, label in graph_nodes.items():
        if is_desired_framenet_node(label):
            return node


def get_argument_node(
        amr: AMR, alignments: List[AMR_Alignment], claim_node: Optional[str]
) -> Optional[str]:
    """Get all argument (claimer) nodes of the claim node"""
    nodes = amr.nodes
    nodes_to_strings = create_node_to_token_dict(amr, alignments)
    amr_dict = create_amr_dict(amr)
    node_args = amr_dict.get(claim_node)
    if node_args:
        claimer_nodes = node_args.get(":ARG0")
        if claimer_nodes:
            claimer_node = claimer_nodes[0]  # only get one
            claimer_label = nodes.get(claimer_node)
            if claimer_label == "person" or claimer_label == "organization":
                return get_full_name_value(amr_dict, nodes_to_strings, claimer_node)
            return get_full_description(
                amr_dict, nodes, nodes_to_strings, claimer_node
            )
