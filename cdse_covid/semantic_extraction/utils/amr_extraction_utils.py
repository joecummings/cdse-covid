"""Collection of AMR extraction utils."""
from collections import defaultdict
import logging
import re
import string
from typing import Any, Dict, List, MutableMapping, Optional

from amr_utils.alignments import AMR_Alignment  # pylint: disable=import-error
from amr_utils.amr import AMR  # pylint: disable=import-error
from nltk.corpus import stopwords

from cdse_covid.claim_detection.claim import Claim, create_id
from cdse_covid.semantic_extraction.mentions import XVariable

PROPBANK_PATTERN = r"[a-z, -]*-[0-9]{2}"  # e.g. have-name-91
PRONOUNS = {
    "he",
    "she",
    "it",
    "one",
    "they",
    "them",
    "we",
    "our",
    "you",
    "your",
    "ours",
    "her",
    "him",
    "I",
    "me",
}
STOP_WORDS = set(stopwords.words("english")).union({"like"}) - PRONOUNS


def get_full_name_value(
    amr_dict: Dict[str, Dict[str, List[str]]], nodes_to_strings: Dict[str, str], named_node: str
) -> Optional[str]:
    """Get the full name of a named_node."""
    name_nodes = amr_dict[named_node].get(":name")
    if name_nodes:
        name_strings = list(
            filter(None, [nodes_to_strings.get(name_node) for name_node in name_nodes])
        )
        return " ".join(name_strings)
    return None


def get_full_description(
    amr_dict: Dict[str, Dict[str, List[str]]],
    nodes_to_labels: Dict[str, str],
    nodes_to_strings: Dict[str, str],
    tokens_to_indices: Dict[str, int],
    focus_node: str,
    ignore_focus_node: bool = False,
) -> str:
    """Returns the 'focus' node text along with any modifiers.

    If `ignore_focus_node` is True, it will not count the original token(s)
    associated with that node; useful in cases where we only care about the modifiers.

    An argument like "salt water" will be represented as
    ARG0: "water"
    '--> mod: "salt"

    If the focus node label is a PropBank frame ("drink-01"),
    it will attempt to get the text of its "patient" argument.

    The final string will be ordered based on their index in the soruce text.
    Any modifiers found that are not adjacent to the focus text or other
    modifiers will be cut.
    """
    descr_strings = {}

    def add_strings_to_full_description(string_to_add: str) -> None:
        string_tokens = string_to_add.split(" ")
        for token in string_tokens:
            token_idx = tokens_to_indices.get(token)
            if token_idx is not None:  # possible index 0 is a falsey value
                descr_strings[token_idx] = token

    focus_string = nodes_to_strings.get(focus_node)
    if not focus_string:
        logging.warning(
            "Couldn't get source text of AMR node %s.\n" "nodes_to_strings: %s",
            focus_node,
            nodes_to_strings,
        )
        return ""
    if re.match(PROPBANK_PATTERN, nodes_to_labels[focus_node]):
        node_args = amr_dict[focus_node]
        for arg_role, arg_node_list in node_args.items():
            # Only check ARG1 to avoid grabbing extraneous arguments
            if arg_role == ":ARG1":
                arg_description = get_full_description(
                    amr_dict, nodes_to_labels, nodes_to_strings, tokens_to_indices, arg_node_list[0]
                )
                if arg_description:
                    add_strings_to_full_description(arg_description)
                    break
    else:

        def add_sentence_text_to_variable(arg_role: str) -> None:
            arg_list = amr_dict[focus_node].get(arg_role)
            if arg_list:
                # Only use the first one
                first_arg_option = nodes_to_strings.get(arg_list[0])
                if first_arg_option:
                    add_strings_to_full_description(first_arg_option)
                elif not first_arg_option:
                    logging.warning(
                        "Couldn't get source text of AMR node %s.\n" "nodes_to_strings: %s",
                        arg_list[0],
                        nodes_to_strings,
                    )

        # First check for :mods
        mods_of_focus_node = amr_dict[focus_node].get(":mod")
        if mods_of_focus_node:
            for mod in mods_of_focus_node:
                mod_string = nodes_to_strings.get(mod)
                if mod_string:
                    add_strings_to_full_description(mod_string)
                else:
                    logging.warning(
                        "Couldn't get source text of AMR node %s.\n" "nodes_to_strings: %s",
                        mod,
                        nodes_to_strings,
                    )

        # Other mods come from :consists-of and :ARG1-of
        # and they tend to precede :mods in word order
        add_sentence_text_to_variable(":consist-of")
        add_sentence_text_to_variable(":ARG1-of")

        op_of = amr_dict[focus_node].get(":op1")
        # If no mods have been found yet, try op1
        if op_of and not descr_strings:
            op_of_string = nodes_to_strings.get(op_of[0])
            if op_of_string:
                add_strings_to_full_description(op_of_string)
            else:
                logging.warning(
                    "Couldn't get source text of AMR node %s.\n" "nodes_to_strings: %s",
                    op_of,
                    nodes_to_strings,
                )
    # Sort and join strings
    if not ignore_focus_node:
        add_strings_to_full_description(focus_string)
    descr_string_list = [descr_strings[i] for i in sorted(descr_strings)]

    # Handle "X and Y" claimer values
    if len(descr_string_list) == 3 and descr_string_list[1] == "and":
        # Select only the first claimer as this causes less confusion finding qnodes
        descr_string_list = descr_string_list[:1]

    # Eliminate tokens with an index not adjacent to the focus token(s)
    if len(descr_string_list) > 1 and not ignore_focus_node:
        first_focus_token = focus_string.split(" ")[0]
        first_focus_token_idx = tokens_to_indices[first_focus_token]
        descr_list_idx = descr_string_list.index(first_focus_token)
        i = 0
        to_remove = []
        for idx, descr_token in sorted(descr_strings.items()):
            # Compare mod-focus distances between the source text
            # and the calculated string -- they should be the same.
            if abs(idx - first_focus_token_idx) != abs(i - descr_list_idx):
                to_remove.append(descr_token)
            i += 1
        for faraway_token in to_remove:
            descr_string_list.remove(faraway_token)
    return " ".join(descr_string_list)


def create_node_to_token_dict(amr: AMR, alignments: List[AMR_Alignment]) -> Dict[str, str]:
    """Creates a MutableMapping between AMR graph nodes and tokens from the source sentence."""
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
                if token_text not in string.punctuation and token_text != "<ROOT>":
                    nodes_to_token_lists[node].append(token_text)
    return {node: " ".join(token_list) for node, token_list in nodes_to_token_lists.items()}


def create_tokens_to_indices(amr_tokens: List[str]) -> Dict[str, int]:
    """Create a mapping of tokens to their indices in a sentence or claim."""
    return {token: index for index, token in enumerate(amr_tokens)}


def remove_preceding_trailing_stop_words(text: Optional[str]) -> Optional[str]:
    """Remove trailing stop words and punctuation from mention text."""
    if not text:
        return None
    text_tokens = text.split(" ")
    first_nonstop_idx = -1
    last_nonstop_idx = -1
    for i, token in enumerate(text_tokens):
        if (
            first_nonstop_idx == -1
            and token.lower() not in STOP_WORDS
            and token not in string.punctuation
        ):
            first_nonstop_idx = i
        if (
            first_nonstop_idx > -1
            and token.lower() not in STOP_WORDS
            and token not in string.punctuation
        ):
            last_nonstop_idx = i
    clipped_text = text_tokens[first_nonstop_idx : last_nonstop_idx + 1]
    return " ".join(clipped_text)


def create_x_variable(text: Optional[str], claim: Claim, tokenizer: Any) -> Optional[XVariable]:
    """Return an X-Variable object using the variable text and its claim data."""
    if text:
        # Remove trailing stop words
        final_text = remove_preceding_trailing_stop_words(text)
        if final_text:
            print("We're about to create the x-variable...")
            return XVariable(
                mention_id=create_id(),
                text=final_text,
                doc_id=claim.doc_id,
                span=claim.get_offsets_for_text(final_text, tokenizer),
            )
    return None


def identify_x_variable_covid(
    amr: AMR, alignments: List[AMR_Alignment], claim: Claim, tokenizer: Any
) -> Optional[XVariable]:
    """Use the AMR graph of the claim to identify the X variable given the template."""
    claim_template = claim.claim_template
    if not claim_template:
        return None

    place_variables = {"facility", "location", "place"}  # from the templates
    place_types = {"city", "state", "country", "continent"}
    amr_dict = amr.edge_mapping()
    nodes_to_labels = amr.nodes
    nodes_to_source_strings = create_node_to_token_dict(amr, alignments)
    tokens_to_indices = create_tokens_to_indices(amr.tokens)

    # For claims with "location-X" templates, locate a FAC/GPE/LOC
    if any(f"{place_term}-X" in claim_template for place_term in place_variables):
        for parent, role, child in amr.edges:
            child_label = nodes_to_labels.get(child)
            # Not all locations get the :location role label
            if role == ":location" or role == ":source" or child_label in place_types:
                location_name = get_full_name_value(amr_dict, nodes_to_source_strings, child)
                return (
                    create_x_variable(location_name, claim, tokenizer)
                    if location_name
                    else create_x_variable(
                        get_full_description(
                            amr_dict,
                            nodes_to_labels,
                            nodes_to_source_strings,
                            tokens_to_indices,
                            child,
                        ),
                        claim,
                        tokenizer,
                    )
                )

    # For "person-X" templates, locate a person
    if "person-X" in claim_template:
        for parent, role, child in amr.edges:
            child_label = nodes_to_labels.get(child)
            if child_label == "person":
                person_name = get_full_name_value(amr_dict, nodes_to_source_strings, child)
                return (
                    create_x_variable(person_name, claim, tokenizer)
                    if person_name
                    else create_x_variable(child_label, claim, tokenizer)
                )
    if claim_template.endswith("is X"):
        # In such cases, X is usually the root of the claim graph.
        return create_x_variable(
            get_full_description(
                amr_dict, nodes_to_labels, nodes_to_source_strings, tokens_to_indices, amr.root
            ),
            claim,
            tokenizer,
        )
    if claim_template.startswith("X was the target"):
        # For the template concerning the target of the coronavirus,
        # find the ARG1 of "target-01"
        for parent, role, child in amr.edges:
            if nodes_to_labels[parent] == "target-01" and role == ":ARG1":
                target_name = get_full_name_value(amr_dict, nodes_to_source_strings, child)
                return (
                    create_x_variable(target_name, claim, tokenizer)
                    if target_name
                    else create_x_variable(
                        get_full_description(
                            amr_dict,
                            nodes_to_labels,
                            nodes_to_source_strings,
                            tokens_to_indices,
                            child,
                            ignore_focus_node=True,
                        ),
                        claim,
                        tokenizer,
                    )
                )
    if "X negative effect" in claim_template:
        # This concerns negative effects of wearing masks;
        # Find the mod(s) of affect-01
        for parent, role, child in amr.edges:
            if nodes_to_labels[parent] == "affect-01":
                return create_x_variable(
                    get_full_description(
                        amr_dict,
                        nodes_to_labels,
                        nodes_to_source_strings,
                        tokens_to_indices,
                        parent,
                        ignore_focus_node=True,
                    ),
                    claim,
                    tokenizer,
                )
    if "Government-X" in claim_template:
        # In these graphs, the GPE of "government" is not a mod,
        # so we append the GPE with "government" if it is a token in the sentence.
        add_gov_token = "government" in amr.tokens
        for parent, role, child in amr.edges:
            if nodes_to_labels[parent] == "government-organization":
                # try up to two steps down
                full_name = None
                if nodes_to_labels[child] in place_types:
                    full_name = get_full_name_value(amr_dict, nodes_to_source_strings, child)
                else:
                    gov_args = amr_dict[child]
                    for values in gov_args.values():
                        for value in values:
                            if nodes_to_labels[value] in place_types:
                                full_name = get_full_name_value(
                                    amr_dict, nodes_to_source_strings, value
                                )
                if full_name and add_gov_token:
                    return create_x_variable(full_name + " government", claim, tokenizer)
                return create_x_variable(full_name, claim, tokenizer)
            # Cover the case where the location name is used to represent its
            # government. We assume that in most cases, the first ARG0 that is also a
            # GPE will be the government in question.
            elif role == ":ARG0" and nodes_to_labels[child] in place_types:
                return create_x_variable(
                    get_full_name_value(amr_dict, nodes_to_source_strings, child), claim, tokenizer
                )
    # For "date-X" templates, return the date-entity
    if "date-X" in claim_template:
        for node, node_label in nodes_to_labels.items():
            if node_label == "date-entity":
                return create_x_variable(
                    get_full_description(
                        amr_dict, nodes_to_labels, nodes_to_source_strings, tokens_to_indices, node
                    ),
                    claim,
                    tokenizer,
                )
    # This covers "treatment-X" template cases
    if "Treatment-X" in claim_template or "effective treatment" in claim_template:
        for parent, role, child in amr.edges:
            if (
                (mislablled_treatment(nodes_to_labels, parent, role))
                or (treatment_in_arg3(nodes_to_labels, parent, role))
                or (shortens_infection(nodes_to_labels, parent, role))
                or (prevents_death(nodes_to_labels, parent, role))
                or (treatment_is_approved(nodes_to_labels, parent, role))
            ):
                return create_x_variable(
                    get_full_description(
                        amr_dict, nodes_to_labels, nodes_to_source_strings, tokens_to_indices, child
                    ),
                    claim,
                    tokenizer,
                )
    if "medication X" in claim_template:
        # Concerns safe medication being unsafe for COVID-19 patients;
        # look for safe-01
        for parent, role, child in amr.edges:
            if nodes_to_labels[parent] == "safe-01" and role == ":ARG1":
                return create_x_variable(
                    get_full_description(
                        amr_dict, nodes_to_labels, nodes_to_source_strings, tokens_to_indices, child
                    ),
                    claim,
                    tokenizer,
                )
    if "Animal-X" in claim_template:
        # We're going to look at the root arguments for this.
        # The only "animal" template describes an animal "being involved"
        # with the origin of COVID-19, so we'll try to look at ARG1 of the root.
        root_args = amr_dict[amr.root]
        arg1_values = root_args.get(":ARG1")
        if arg1_values:
            # Get only one
            return create_x_variable(
                get_full_description(
                    amr_dict,
                    nodes_to_labels,
                    nodes_to_source_strings,
                    tokens_to_indices,
                    arg1_values[0],
                ),
                claim,
                tokenizer,
            )
    # The next two conditions are meant to cover all other templates.
    # If X is the first in the template, it implies that it serves the agent role
    if claim_template[0] == "X":
        # The "agent" of cure-01 is ARG3
        agent_role = ":ARG3" if claim_template == "X cures COVID-19" else ":ARG0"
        for parent, role, child in amr.edges:
            if parent == amr.root and role == agent_role:
                agent_name = get_full_name_value(amr_dict, nodes_to_source_strings, child)
                if agent_name:
                    return create_x_variable(agent_name, claim, tokenizer)
                return create_x_variable(
                    get_full_description(
                        amr_dict, nodes_to_labels, nodes_to_source_strings, tokens_to_indices, child
                    ),
                    claim,
                    tokenizer,
                )
    # Likewise, if X is the last item in the template,
    # it implies a patient role
    if claim_template[-1] == "X":
        for parent, role, child in amr.edges:
            if parent == amr.root and role == ":ARG1":
                patient_name = get_full_name_value(amr_dict, nodes_to_source_strings, child)
                if patient_name:
                    return create_x_variable(patient_name, claim, tokenizer)
                return create_x_variable(
                    get_full_description(
                        amr_dict, nodes_to_labels, nodes_to_source_strings, tokens_to_indices, child
                    ),
                    claim,
                    tokenizer,
                )
    return None


def treatment_is_approved(
    nodes_to_labels: MutableMapping[str, Any], parent: str, role: str
) -> bool:
    """If treatment is in the ARG1 spot."""
    return nodes_to_labels[parent] == "approve-01" and role == ":ARG1"


def prevents_death(nodes_to_labels: MutableMapping[str, Any], parent: str, role: str) -> bool:
    """If treatment is in ARG0 spot."""
    return nodes_to_labels[parent] == "prevent-01" and role == ":ARG0"


def shortens_infection(nodes_to_labels: MutableMapping[str, Any], parent: str, role: str) -> bool:
    """If treatment is in ARG0 and verb is shorten."""
    return nodes_to_labels[parent] == "shorten-01" and role == ":ARG0"


def treatment_in_arg3(nodes_to_labels: MutableMapping[str, Any], parent: str, role: str) -> bool:
    """If treatment is in ARG3 spot."""
    return nodes_to_labels[parent] == "treat-03" and role == ":ARG3"


def mislablled_treatment(nodes_to_labels: MutableMapping[str, Any], parent: str, role: str) -> bool:
    """Often the system mislabels treatments as ARG1 (which is supposed to be the "patient"), so we'll check it anyway."""
    return nodes_to_labels[parent] == "treat-03" and role == ":ARG1"


def identify_x_variable(
    amr: AMR,
    alignments: List[AMR_Alignment],
    claim: Claim,
    claim_ents: Dict[str, str],
    claim_pos: Dict[str, str],
    tokenizer: Any,
) -> Optional[XVariable]:
    """Use the AMR graph of the claim to identify the X variable given the claim text.

    An alternative to `identify_x_variable_covid` that doesn't rely on the templates
    of our COVID-19 domain.
    """
    place_types = {"city", "state", "country", "continent"}
    amr_dict = amr.edge_mapping()
    nodes_to_labels = amr.nodes
    nodes_to_source_strings = create_node_to_token_dict(amr, alignments)
    tokens_to_indices = create_tokens_to_indices(amr.tokens)

    # First use entity labels as clues for what the X-variable is
    for _, label in claim_ents.items():
        if label == "NORP":
            # A nationality may hint at the variable
            for parent, role, child in amr.edges:
                parent_label = nodes_to_labels.get(parent)
                child_label = nodes_to_labels.get(child)
                # Check if it's a government organization
                if parent_label == "government-organization":
                    add_gov_token = "government" in amr.tokens
                    # try up to two steps down
                    full_name = None
                    if child_label in place_types:
                        full_name = get_full_name_value(amr_dict, nodes_to_source_strings, child)
                    else:
                        gov_args = amr_dict[child]
                        for values in gov_args.values():
                            for value in values:
                                if nodes_to_labels[value] in place_types:
                                    full_name = get_full_name_value(
                                        amr_dict, nodes_to_source_strings, value
                                    )
                    if full_name and add_gov_token:
                        return create_x_variable(full_name + " government", claim, tokenizer)
                    return create_x_variable(full_name, claim, tokenizer)
                if parent_label in place_types:
                    return create_x_variable(
                        get_full_description(
                            amr_dict,
                            nodes_to_labels,
                            nodes_to_source_strings,
                            tokens_to_indices,
                            parent,
                        ),
                        claim,
                        tokenizer,
                    )
                elif child_label in place_types:
                    # If the nationality is a mod, check the parent
                    if claim_pos.get(nodes_to_source_strings[child]) == "ADJ":
                        return create_x_variable(
                            get_full_description(
                                amr_dict,
                                nodes_to_labels,
                                nodes_to_source_strings,
                                tokens_to_indices,
                                parent,
                            ),
                            claim,
                            tokenizer,
                        )
                    else:
                        return create_x_variable(
                            get_full_description(
                                amr_dict,
                                nodes_to_labels,
                                nodes_to_source_strings,
                                tokens_to_indices,
                                child,
                            ),
                            claim,
                            tokenizer,
                        )
        if label in ["PERSON", "ORG"]:
            # If a PERSON/ORG is detected, get the full name
            for parent, role, child in amr.edges:
                child_label = nodes_to_labels.get(child)
                if child_label in ["person", "organization"]:
                    person_name = get_full_name_value(amr_dict, nodes_to_source_strings, child)
                    return (
                        create_x_variable(person_name, claim, tokenizer)
                        if person_name
                        else create_x_variable(child_label, claim, tokenizer)
                    )

    # Next, simply look for a location
    for parent, role, child in amr.edges:
        parent_label = nodes_to_labels.get(parent)
        child_label = nodes_to_labels.get(child)
        # Not all locations get the :location role label
        if role == ":location" or role == ":source" or child_label in place_types:
            location_name = get_full_name_value(amr_dict, nodes_to_source_strings, child)
            if location_name:
                return create_x_variable(location_name, claim, tokenizer)
            else:
                return create_x_variable(
                    get_full_description(
                        amr_dict, nodes_to_labels, nodes_to_source_strings, tokens_to_indices, child
                    ),
                    claim,
                    tokenizer,
                )
        # If there is a date-entity in the AMR graph, that may be the X-variable
        elif parent_label == "date-entity":
            return create_x_variable(
                get_full_description(
                    amr_dict, nodes_to_labels, nodes_to_source_strings, tokens_to_indices, parent
                ),
                claim,
                tokenizer,
            )
    return None
