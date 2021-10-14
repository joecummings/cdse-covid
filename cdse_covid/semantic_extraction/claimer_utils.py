from collections import defaultdict
from amr_utils.amr import AMR
from amr_utils.propbank_frames import propbank_frames_dictionary
from typing import Dict, List, Optional
from nltk.stem import WordNetLemmatizer

LEMMATIZER = WordNetLemmatizer()

CLAIMER_ROLES = {
    "admitter",
    "arguer",
    "asserter",
    "claimer",
    "sayer",
    "speaker",
    "researcher"
}


def identify_claimer(amr: AMR) -> List[str]:
    """Identify the claimer of the span."""
    if not amr:
        return ""
    # Find the claim node by using one of two rules:
    # 1) Search for the Statement node of the claim by working up the graph
    # 2) If (1) fails, find the first Statement node in the graph
    claim_node = get_claim_node(amr.tokens, amr)
    return get_argument_node(amr, claim_node)


def get_claim_node(claim_tokens: List[str], amr: AMR) -> Optional[str]:
    """Get the head node of the claim"""
    graph_nodes = amr.nodes
    for token in claim_tokens:
        for node, label in graph_nodes.items():
            # We're hoping that at least one nominal/verbial lemma is found
            if (
                LEMMATIZER.lemmatize(token, pos="n") == label
                or LEMMATIZER.lemmatize(token, pos="v") == label
            ):
                return get_claim_node_from_token(node, graph_nodes, amr.edges, 0)
    return search_for_claim_node(graph_nodes)


def is_statement_node(node_label: str) -> bool:
    """Determine if the node under investigation represents a statement event"""
    if node_label in propbank_frames_dictionary:
        for claimer_role in CLAIMER_ROLES:
            if claimer_role in propbank_frames_dictionary[node_label].lower():
                return True
    return False


def get_claim_node_from_token(node, node_dict, edges, i) -> Optional[str]:
    """Fetch the claim node by traveling up from a child node"""
    for parent_node, _, arg_node in edges:
        if arg_node == node:
            # Check if the parent is a claim node
            parent_label = node_dict[parent_node]
            if is_statement_node(parent_label):
                return parent_node
            if i == 10:
                break
            return get_claim_node_from_token(parent_node, node_dict, edges, i + 1)
    return search_for_claim_node(node_dict)


def search_for_claim_node(graph_nodes) -> Optional[str]:
    """
    Rule #2: try finding the statement node by reading through all nodes
    and returning the first match
    """
    for node, label in graph_nodes.items():
        if is_statement_node(label):
            return node
    return None


def create_amr_dict(amr: AMR) -> Dict[str, Dict[str, str]]:
    amr_dict = defaultdict(dict)
    for parent, role, arg in amr.edges:
        amr_dict[parent][role] = arg
    return amr_dict


def get_argument_node(amr: AMR, claim_node: Optional[str]) -> List[str]:
    """Get all argument (claimer) nodes of the claim node"""
    claimers = set()
    nodes = amr.nodes
    amr_dict = create_amr_dict(amr)
    node_args = amr_dict.get(claim_node)
    if node_args:
        claimer_node = node_args.get(":ARG0")
        if not claimer_node:
            return list(claimers)
        claimer_label = nodes[claimer_node]
        if claimer_label == "person":
            name = get_claimer_name(amr_dict, nodes, claimer_node)
            if len(name) > 0:
                claimers.add(f"'{name}'")
        elif claimer_label == "and":
            for role, arg_node in amr_dict[claimer_node].items():
                co_claimer_label = nodes[arg_node]
                if co_claimer_label == "person":
                    name = get_claimer_name(amr_dict, nodes, arg_node)
                    if len(name) > 0:
                        claimers.add(f"'{name}'")
                else:
                    claimers.add(co_claimer_label)
        else:
            claimers.add(claimer_label)
    return list(claimers)


def get_claimer_name(
    amr_dict: Dict[str, Dict[str, str]], node_dict: Dict[str, str], person_node: str
) -> str:
    name_strings = []
    name_node = amr_dict[person_node].get(":name")
    if name_node:
        person_args = amr_dict[name_node]
        for role, node in person_args.items():
            if role.startswith(":op"):
                name_strings.append(node_dict[node].strip('"'))
    return " ".join(name_strings)
