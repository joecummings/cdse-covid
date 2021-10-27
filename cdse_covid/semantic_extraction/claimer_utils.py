import string
from collections import defaultdict

from amr_utils.alignments import AMR_Alignment
from amr_utils.amr import AMR
from nltk.corpus import framenet
from typing import Dict, List, Optional
from nltk.stem import WordNetLemmatizer
import re

LEMMATIZER = WordNetLemmatizer()


framenet_concepts = ["Statement", "Reasoning"]
FRAMENET_VERBS = set()
for concept in framenet_concepts:
    frame = framenet.frame_by_name(concept)
    lex_units = frame.get("lexUnit")
    for verb in lex_units.keys():
        word, pos = verb.split(".")
        if pos == "v":
            word = word.replace(" ", "-") # Account for multiple words
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
    verb = node_label.rsplit("-", 1)[0] # E.g. origin from origin-01
    return verb in FRAMENET_VERBS


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


def create_amr_dict(amr: AMR) -> Dict[str, Dict[str, List[str]]]:
    # {"parent": {"role1": ["child1"], "role2": ["child2", "child3"]}, ...}
    amr_dict = defaultdict(lambda: defaultdict(list))
    for parent, role, arg in amr.edges:
        amr_dict[parent][role].append(arg)
    return amr_dict


def create_node_to_token_dict(
    amr: AMR, alignments: List[AMR_Alignment]
) -> Dict[str, str]:
    """
    Creates a mapping between AMR graph nodes and tokens from the source sentence
    """
    amr_tokens = amr.tokens
    # Use this dict to get the list of tokens first
    nodes_to_token_lists = defaultdict(list)
    for alignment in alignments:
        # Make sure all of the 'token' values are lists
        preprocessed_alignment = fix_alignment(alignment)
        alignment_dict = preprocessed_alignment.to_json(amr)
        nodes = alignment_dict["nodes"]
        tokens = alignment_dict["tokens"]
        for node in nodes:
            for token in tokens:
                token_text = amr_tokens[int(token)]
                # ignore punctuation
                if token_text not in string.punctuation:
                    nodes_to_token_lists[node].append(token_text)
    # Then map from nodes to token strings
    nodes_to_strings = {
        node: " ".join(token_list)
        for node, token_list in nodes_to_token_lists.items()
    }

    return nodes_to_strings


def fix_alignment(alignment: AMR_Alignment) -> AMR_Alignment:
    """Make sure all of the 'token' values are lists"""
    alignment_tokens = alignment.tokens
    if type(alignment_tokens) == str:
        # The value will look like '2-3'
        first_token, last_token = alignment_tokens.split("-")
        try:
            first_tok_int = int(first_token)
            last_tok_int = int(last_token)
        except ValueError:
            raise RuntimeError(
                "Looks like something is wrong with the token value conversion."
            )
        token_range = range(first_tok_int, last_tok_int+1)
        alignment.tokens = [tok_index for tok_index in token_range]
    return alignment


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
            if claimer_label == "person":
                return get_full_name_value(amr_dict, nodes_to_strings, claimer_node)
            return get_full_description(amr_dict, nodes_to_strings, claimer_node)


def get_full_name_value(
    amr_dict: Dict[str, Dict[str, List[str]]],
    nodes_to_strings: Dict[str, str],
    named_node: str
) -> Optional[str]:
    name_nodes = amr_dict[named_node].get(":name")
    if name_nodes:
        name_strings = [
            nodes_to_strings[name_node]
            for name_node in name_nodes
        ]
        return " ".join(name_strings)


def get_full_description(
    amr_dict: Dict[str, Dict[str, List[str]]], nodes_to_strings: Dict[str, str], focus_node: str
) -> str:
    """
    Returns the 'focus' node text along with any modifiers

    An argument like "salt water" will be represented as
    ARG0: "water"
    '--> mod: "salt"
    """
    descr_strings = [nodes_to_strings[focus_node]]
    # First check for :mods
    mods_of_focus_node = amr_dict[focus_node].get(":mod")
    if mods_of_focus_node:
        for mod in mods_of_focus_node:
            descr_strings.insert(0, nodes_to_strings[mod])
    # Other mods come from :ARG1-of
    # and they precede :mods in word order
    arg1s_of = amr_dict[focus_node].get(":ARG1-of")
    if arg1s_of:
        # Only use the first one
        descr_strings.insert(0, nodes_to_strings[arg1s_of[0]])
    return " ".join(descr_strings)


def identify_x_variable(amr: AMR, alignments: List[AMR_Alignment], claim_template: str) -> Optional[str]:
    """
    Use the AMR graph of the claim to identify the X variable given the template
    todo: finish adding rules for all possible templates
    """
    place_variables = {"facility", "location", "place"}
    place_types = {"city", "country"}
    amr_dict = create_amr_dict(amr)
    node_dict = amr.nodes
    nodes_to_source_strings = create_node_to_token_dict(amr, alignments)

    if any(f"{place_term}-X" in claim_template for place_term in place_variables):
        # Locate a FAC/GPE/LOC
        for parent, role, child in amr.edges:
            child_label = node_dict.get(child)
            # todo: find all possible location labels
            if role == "location" or any(
                    child_label == place_type for place_type in place_types
            ):
                location_name = nodes_to_source_strings.get(child)
                return location_name if location_name else child_label
    if "person-X" in claim_template:
        # Locate a person
        for parent, role, child in amr.edges:
            child_label = node_dict.get(child)
            if child_label == "person":
                person_name = get_full_name_value(
                    amr_dict, nodes_to_source_strings, child
                )
                return person_name if person_name else child_label
    if claim_template == "SARS-CoV-2 is X":
        # In such cases, X is usually the root.
        return get_full_description(amr_dict, nodes_to_source_strings, amr.root)
    # Maybe too simple a rule: if X is the first in the template,
    # it implies that it serves the agent role
    if claim_template[0] == "X":
        "Let's look for an agent!"
        # The "agent" of cure-01 is ARG3
        if claim_template == "X cures COVID-19":
            agent_role = ":ARG3"
        else:
            agent_role = ":ARG0"
        for parent, role, child in amr.edges:
            if parent == amr.root and role == agent_role:
                agent_name = get_full_name_value(
                    amr_dict, nodes_to_source_strings, child
                )
                if agent_name:
                    return agent_name
                return get_full_description(amr_dict, nodes_to_source_strings, child)
    # Likewise, if X is the last item in the template,
    # it implies a patient role
    if claim_template[-1] == "X":
        for parent, role, child in amr.edges:
            if parent == amr.root and role == ":ARG1":
                patient_name = get_full_name_value(
                    amr_dict, nodes_to_source_strings, child
                )
                if patient_name:
                    return patient_name
                return get_full_description(amr_dict, nodes_to_source_strings, child)
