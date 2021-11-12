from dataclasses import dataclass
from pathlib import Path
import re
import string
from collections import defaultdict
from typing import List, Optional, Dict, Union

from amr_utils.alignments import AMR_Alignment
from amr_utils.amr import AMR
from amr_utils.amr_readers import AMR_Reader


ALIGNMENTS_TYPE = Dict[Union[List[str], str], Union[List[AMR_Alignment], list]]
AMR_READER = AMR_Reader()

@dataclass
class LocalAMR:
    doc_id: str
    graph: List[AMR]
    alignments: Optional[ALIGNMENTS_TYPE] = None


def load_amr_from_text_file(
        amr_file: Path, output_alignments: bool = False
) -> LocalAMR:
    """
    Reads a document of AMR graphs and returns an AMR graph.
    If `output_alignments` is True, it will also return
    the alignment data of that graph.
    """
    if output_alignments:
        amr, alignments = AMR_READER.load(amr_file, remove_wiki=True, output_alignments=True)
        return LocalAMR(amr_file.stem, amr, alignments)
    amr = AMR_READER.load(amr_file, remove_wiki=True)
    return LocalAMR(amr_file.stem, amr)

PROPBANK_PATTERN = r"[a-z]*-[0-9]{2}"


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
    amr_dict: Dict[str, Dict[str, List[str]]],
        nodes_to_labels: Dict[str, str],
        nodes_to_strings: Dict[str, str],
        focus_node: str,
        ignore_focus_node: bool = False,
) -> str:
    """
    Returns the 'focus' node text along with any modifiers

    If `ignore_focus_node` is True, it will not count the original token(s)
    associated with that node; useful in cases where we only care about the modifiers.

    An argument like "salt water" will be represented as
    ARG0: "water"
    '--> mod: "salt"

    If the focus node label is a PropBank frame ("drink-01"),
    it will attempt to get the text of its "patient" argument.

    The resulting string will be in this order:
    <ARG1-of> <consist-of> <mod>* <focus_node> <op1> <ARG1>
    """
    descr_strings = []
    focus_string = nodes_to_strings[focus_node]
    if re.match(PROPBANK_PATTERN, nodes_to_labels[focus_node]):
        node_args = amr_dict[focus_node]
        for arg_role, arg_node_list in node_args.items():
            # Only check ARG1 to avoid grabbing extraneous arguments
            if arg_role == ":ARG1":
                arg_description = get_full_description(
                    amr_dict, nodes_to_labels, nodes_to_strings, arg_node_list[0]
                )
                if arg_description:
                    descr_strings.append(arg_description)
                    break
        # Duplicate tokens are naturally uncommon, so avoid adding them
        # since they are probably due to a cyclical AMR graph
        if focus_string not in descr_strings:
            descr_strings.insert(0, focus_string)
    else:
        def add_sentence_text_to_variable(arg_role: str) -> None:
            arg_list = amr_dict[focus_node].get(arg_role)
            if arg_list:
                # Only use the first one
                first_arg_option = nodes_to_strings[arg_list[0]]
                if first_arg_option not in descr_strings:
                    descr_strings.insert(0, first_arg_option)

        # First check for :mods
        mods_of_focus_node = amr_dict[focus_node].get(":mod")
        if mods_of_focus_node:
            for mod in mods_of_focus_node:
                descr_strings.insert(0, nodes_to_strings[mod])

        # Other mods come from :consists-of and :ARG1-of
        # and they tend to precede :mods in word order
        add_sentence_text_to_variable(":consist-of")
        add_sentence_text_to_variable(":ARG1-of")

        op_of = amr_dict[focus_node].get(":op1")
        # If no mods have been found yet, try op1
        descr_string_set = set(descr_strings)
        if op_of and not descr_string_set:
            # Add focus node text before op1
            if not ignore_focus_node and focus_string not in descr_strings:
                descr_strings.append(focus_string)
            descr_strings.append(nodes_to_strings[op_of[0]])
        # Else, just add focus node text here
        elif not ignore_focus_node and focus_string not in descr_strings:
            descr_strings.append(focus_string)
    return " ".join(descr_strings)


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
        alignment_dict = alignment.to_json(amr)
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


def identify_x_variable_covid(
        amr: AMR, alignments: List[AMR_Alignment], claim_template: str
) -> Optional[str]:
    """
    Use the AMR graph of the claim to identify the X variable given the template
    """
    place_variables = {"facility", "location", "place"}  # from the templates
    place_types = {"city", "state", "country", "continent"}
    amr_dict = amr.edge_mapping()
    nodes_to_labels = amr.nodes
    nodes_to_source_strings = create_node_to_token_dict(amr, alignments)

    # For claims with "location-X" templates, locate a FAC/GPE/LOC
    if any(f"{place_term}-X" in claim_template for place_term in place_variables):
        for parent, role, child in amr.edges:
            child_label = nodes_to_labels.get(child)
            # Not all locations get the :location role label
            if role == ":location" or role == ":source" or child_label in place_types:
                location_name = get_full_name_value(
                    amr_dict, nodes_to_source_strings, child
                )
                return location_name if location_name else get_full_description(
                    amr_dict, nodes_to_labels, nodes_to_source_strings, child
                )
    # For "person-X" templates, locate a person
    if "person-X" in claim_template:
        for parent, role, child in amr.edges:
            child_label = nodes_to_labels.get(child)
            if child_label == "person":
                person_name = get_full_name_value(
                    amr_dict, nodes_to_source_strings, child
                )
                return person_name if person_name else child_label
    if claim_template.endswith("is X"):
        # In such cases, X is usually the root of the claim graph.
        return get_full_description(
            amr_dict, nodes_to_labels, nodes_to_source_strings, amr.root
        )
    if claim_template.startswith("X was the target"):
        # For the template concerning the target of the coronavirus,
        # find the ARG1 of "target-01"
        for parent, role, child in amr.edges:
            if nodes_to_labels[parent] == "target-01" and role == ":ARG1":
                target_name = get_full_name_value(
                    amr_dict, nodes_to_source_strings, child
                )
                return target_name if target_name else get_full_description(
                    amr_dict,
                    nodes_to_labels,
                    nodes_to_source_strings,
                    child,
                    ignore_focus_node=True
                )
    if "X negative effect" in claim_template:
        # This concerns negative effects of wearing masks;
        # Find the mod(s) of affect-01
        for parent, role, child in amr.edges:
            if nodes_to_labels[parent] == "affect-01":
                return get_full_description(
                    amr_dict,
                    nodes_to_labels,
                    nodes_to_source_strings,
                    parent,
                    ignore_focus_node=True
                )
    if "Government-X" in claim_template:
        # In these graphs, the GPE of "government" is not a mod,
        # so we append the GPE with "government" if it is a token in the sentence.
        add_gov_token = True if "government" in amr.tokens else False
        for parent, role, child in amr.edges:
            if nodes_to_labels[parent] == "government-organization":
                # try up to two steps down
                full_name = None
                if nodes_to_labels[child] in place_types:
                    full_name = get_full_name_value(
                        amr_dict, nodes_to_source_strings, child
                    )
                else:
                    gov_args = amr_dict[child]
                    for values in gov_args.values():
                        for value in values:
                            if nodes_to_labels[value] in place_types:
                                full_name = get_full_name_value(
                                    amr_dict, nodes_to_source_strings, value
                                )
                if full_name and add_gov_token:
                    return full_name + " government"
                return full_name
            # Cover the case where the location name is used to represent its
            # government. We assume that in most cases, the first ARG0 that is also a
            # GPE will be the government in question.
            elif role == ":ARG0" and nodes_to_labels[child] in place_types:
                return get_full_name_value(amr_dict, nodes_to_source_strings, child)
    # For "date-X" templates, return the date-entity
    if "date-X" in claim_template:
        for node, node_label in nodes_to_labels.items():
            if node_label == "date-entity":
                return get_full_description(
                    amr_dict, nodes_to_labels, nodes_to_source_strings, node
                )
    # This covers "treatment-X" template cases
    if "Treatment-X" in claim_template or "effective treatment" in claim_template:
        for parent, role, child in amr.edges:
            # The treatment in treat-03 is supposed to be ARG3
            if (
                    (nodes_to_labels[parent] == "treat-03" and role == ":ARG3") or
                    # Often the system mislabels treatments as ARG1 (which is
                    # supposed to be the "patient"), so we'll check it anyway.
                    (nodes_to_labels[parent] == "treat-03" and role == ":ARG1") or
                    # "treatment-X shortens infection"
                    (nodes_to_labels[parent] == "shorten-01" and role == ":ARG0") or
                    # "treatment-X prevents (chance of) death"
                    (nodes_to_labels[parent] == "prevent-01" and role == ":ARG0") or
                    # "treatment-X received emergency approval"
                    (nodes_to_labels[parent] == "approve-01" and role == ":ARG1")
            ):
                return get_full_description(
                    amr_dict, nodes_to_labels, nodes_to_source_strings, child
                )
    if "medication X" in claim_template:
        # Concerns safe medication being unsafe for COVID-19 patients;
        # look for safe-01
        for parent, role, child in amr.edges:
            if nodes_to_labels[parent] == "safe-01" and role == ":ARG1":
                return get_full_description(
                    amr_dict, nodes_to_labels, nodes_to_source_strings, child
                )
    if "Animal-X" in claim_template:
        # We're going to look at the root arguments for this.
        # The only "animal" template describes an animal "being involved"
        # with the origin of COVID-19, so we'll try to look at ARG1 of the root.
        root_args = amr_dict[amr.root]
        arg1_values = root_args.get(":ARG1")
        if arg1_values:
            # Get only one
            return get_full_description(
                amr_dict, nodes_to_labels, nodes_to_source_strings, arg1_values[0]
            )
    # The next two conditions are meant to cover all other templates.
    # If X is the first in the template, it implies that it serves the agent role
    if claim_template[0] == "X":
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
                return get_full_description(
                    amr_dict, nodes_to_labels, nodes_to_source_strings, child
                )
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
                return get_full_description(
                    amr_dict, nodes_to_labels, nodes_to_source_strings, child
                )


def identify_x_variable(
        amr: AMR,
        alignments: List[AMR_Alignment],
        claim_ents: Dict[str, str],
        claim_pos: Dict[str, str]
) -> Optional[str]:
    """
    Use the AMR graph of the claim to identify the X variable given the claim text

    An alternative to `identify_x_variable_covid` that doesn't rely on the templates
    of our COVID-19 domain
    """
    place_types = {"city", "state", "country", "continent"}
    amr_dict = amr.edge_mapping()
    nodes_to_labels = amr.nodes
    nodes_to_source_strings = create_node_to_token_dict(amr, alignments)

    # First use entity labels as clues for what the X-variable is
    for entity, label in claim_ents.items():
        if label == "NORP":
            # A nationality may hint at the variable
            for parent, role, child in amr.edges:
                parent_label = nodes_to_labels.get(parent)
                child_label = nodes_to_labels.get(child)
                # Check if it's a government organization
                if parent_label == "government-organization":
                    add_gov_token = True if "government" in amr.tokens else False
                    # try up to two steps down
                    full_name = None
                    if child_label in place_types:
                        full_name = get_full_name_value(
                            amr_dict, nodes_to_source_strings, child
                        )
                    else:
                        gov_args = amr_dict[child]
                        for values in gov_args.values():
                            for value in values:
                                if nodes_to_labels[value] in place_types:
                                    full_name = get_full_name_value(
                                        amr_dict, nodes_to_source_strings, value
                                    )
                    if full_name and add_gov_token:
                        return full_name + " government"
                    return full_name
                if parent_label in place_types:
                    return get_full_description(
                        amr_dict, nodes_to_labels, nodes_to_source_strings, parent
                    )
                if child_label in place_types:
                    # If the nationality is a mod, check the parent
                    if claim_pos.get(nodes_to_source_strings[child]) == "ADJ":
                        return get_full_description(
                            amr_dict, nodes_to_labels, nodes_to_source_strings, parent
                        )
                    else:
                        return get_full_description(
                            amr_dict, nodes_to_labels, nodes_to_source_strings, child
                        )
        if label == "PERSON" or label == "ORG":
            # If a PERSON/ORG is detected, get the full name
            for parent, role, child in amr.edges:
                child_label = nodes_to_labels.get(child)
                if child_label == "person" or child_label == "organization":
                    person_name = get_full_name_value(
                        amr_dict, nodes_to_source_strings, child
                    )
                    return person_name if person_name else child_label

    # Next, simply look for a location
    for parent, role, child in amr.edges:
        parent_label = nodes_to_labels.get(parent)
        child_label = nodes_to_labels.get(child)
        # Not all locations get the :location role label
        if role == ":location" or role == ":source" or child_label in place_types:
            location_name = get_full_name_value(
                amr_dict, nodes_to_source_strings, child
            )
            return location_name if location_name else get_full_description(
                amr_dict, nodes_to_labels, nodes_to_source_strings, child
            )
        # If there is a date-entity in the AMR graph, that may be the X-variable
        if parent_label == "date-entity":
            return get_full_description(
                amr_dict, nodes_to_labels, nodes_to_source_strings, parent
            )
