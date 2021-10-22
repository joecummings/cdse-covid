from collections import defaultdict
from amr_utils.amr import AMR
from nltk.corpus import framenet
from typing import Dict, List, Optional
from nltk.stem import WordNetLemmatizer
import re

LEMMATIZER = WordNetLemmatizer()


framenet_concepts = ["Statement", "Reasoning"]
FRAMENET_VERBS = []
for concept in framenet_concepts:
    frame = framenet.frame_by_name(concept)
    lex_units = frame.get("lexUnit")
    for verb in lex_units.keys():
        word, pos = verb.split(".")
        if pos == "v":
            word = word.replace(" ", "-") # Account for multiple words
            FRAMENET_VERBS.append(word)


def identify_claimer(claim_tokens, amr: AMR) -> str:
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
    return get_argument_node(amr, claim_node)


def get_claim_node(claim_tokens: List[str], amr: AMR) -> Optional[str]:
    """Get the head node of the claim."""
    graph_nodes = amr.nodes
    
    for token in claim_tokens:
        token = token.lower() # Make sure the token is lowercased
        for node, label in graph_nodes.items():
            label = re.sub("(-\d*)", "", label) # Remove any PropBank numbering
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
    return any(verb in node_label for verb in FRAMENET_VERBS)


def get_claim_node_from_token(node, node_dict, edges, i) -> Optional[str]:
    """Fetch the claim node by traveling up from a child node."""
    for parent_node, _, arg_node in edges:
        if arg_node == node:
            # Check if the parent is a claim node / Should you stop at the first one?
            parent_label = node_dict[parent_node]
            if is_desired_framenet_node(parent_label):
                return parent_node
            if i == len(node_dict): # Iterated through all nodes without success
                break
            return get_claim_node_from_token(parent_node, node_dict, edges, i + 1)


def search_for_claim_node(graph_nodes) -> Optional[str]:
    """Rule #2: try finding the statement node by reading through all nodes
    and returning the first match.
    """
    for node, label in graph_nodes.items():
        if is_desired_framenet_node(label):
            return node


def create_amr_dict(amr: AMR) -> Dict[str, Dict[str, str]]:
    amr_dict = defaultdict(dict)
    for parent, role, arg in amr.edges:
        amr_dict[parent][role] = arg
    return amr_dict


def get_argument_node(amr: AMR, claim_node: Optional[str]) -> Optional[str]:
    """Get all argument (claimer) nodes of the claim node"""
    nodes = amr.nodes
    amr_dict = create_amr_dict(amr)
    node_args = amr_dict.get(claim_node)
    if node_args:
        claimer_node = node_args.get(":ARG0")
        claimer_label = nodes.get(claimer_node)
        if claimer_label == "person":
            return get_claimer_name(amr_dict, nodes, claimer_node)
        return claimer_label


def get_claimer_name(
    amr_dict: Dict[str, Dict[str, str]], node_dict: Dict[str, str], person_node: str
) -> Optional[str]:
    name_node = amr_dict[person_node].get(":name")
    if name_node:
        person_args = amr_dict[name_node]
        name_strings = [
            node_dict[node].strip('"')
            for role, node in person_args.items()
            if role.startswith(":op")
        ]

        return " ".join(name_strings)
