"""You will need to run this in the transition-amr-parser env."""
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, MutableMapping, Optional, Set, Tuple, Union

from amr_utils.alignments import AMR_Alignment
from amr_utils.amr import AMR  # pylint: disable=import-error
from nltk.stem import WordNetLemmatizer
from spacy.language import Language

from cdse_covid.claim_detection.claim import Claim
from cdse_covid.semantic_extraction.mentions import (
    ClaimArg,
    ClaimEvent,
    ClaimSemantics,
    Mention,
    WikidataQnode,
)
from cdse_covid.semantic_extraction.utils.amr_extraction_utils import (
    PROPBANK_PATTERN,
    STOP_WORDS,
    create_node_to_token_dict,
    remove_preceding_trailing_stop_words,
)
from wikidata_linker.linker import WikidataLinkingClassifier
from wikidata_linker.qnode_mapping_utils import QnodeTables, create_arg_dict
from wikidata_linker.wikidata_linking import CPU, disambiguate_refvar_kgtk, disambiguate_verb_kgtk

MASTER = "master"

PARENT_DIR = Path(__file__).parent


def get_node_from_pb(amr: AMR, pb_label: str) -> str:
    """Get an AMR node from PropBank label."""
    for _id, node in amr.nodes.items():
        if node == pb_label:
            return str(_id)
    return ""


def is_valid_arg_type(arg_type: str) -> bool:
    """Determine if ARG type is in the set we care about."""
    return (
        "ARG" in arg_type or "time" in arg_type or "location" in arg_type or "direction" in arg_type
    )


def determine_argument_from_edge(event_node: str, edge: Tuple[str, str, str]) -> Optional[str]:
    """Determine which node in an edge serves as the argument.

    If the argument role has the suffix "-of" and the node is in the child
    position, the argument is the parent node.
    Vice versa if it is a regular argument role and the node is the parent.
    Otherwise, a node -> argument relation is not in this edge.
    """
    if edge[2] == event_node and edge[1].endswith("-of"):
        return edge[0]
    elif edge[0] == event_node and not edge[1].endswith("-of"):
        return edge[2]
    return None


def get_framenet_arg_role(pb_arg_role_label: str) -> str:
    """From a PropBank argument role label, return the corresponding FrameNet one."""
    formatted_arg_type = pb_arg_role_label.replace(":", "").replace("-of", "")
    if formatted_arg_type[0] == "A":
        # e.g. ARG1 --> A1
        return formatted_arg_type.replace("RG", "")
    elif formatted_arg_type in {"location", "direction"}:
        # e.g. location --> loc
        return formatted_arg_type[:3]
    else:
        # time doesn't change
        return formatted_arg_type


def get_all_labeled_args(
    amr: AMR, alignments: List[AMR_Alignment], node: str, qnode_args: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Get all labeled args from AMR node."""
    potential_args = amr.get_edges_for_node(node)
    node_labels_to_tokens = create_node_to_token_dict(amr, alignments)
    labeled_args = {}

    for arg in potential_args:
        if is_valid_arg_type(arg[1]):
            arg_node = determine_argument_from_edge(node, arg)
            if arg_node:
                framenet_arg = get_framenet_arg_role(arg[1])
                if qnode_args.get(framenet_arg):
                    node_label = amr.nodes[arg_node]
                    trimmed_node_label = remove_preceding_trailing_stop_words(node_label)
                    tokens_of_node = remove_preceding_trailing_stop_words(
                        node_labels_to_tokens.get(arg_node)
                    )
                    if tokens_of_node:
                        labeled_args[framenet_arg] = tokens_of_node
                    elif trimmed_node_label:
                        labeled_args[framenet_arg] = node_label.rsplit("-", 1)[0]

    return labeled_args


def is_pb_label(label: str) -> bool:
    """Determine whether the AMR graph label is a PropBank frame label."""
    return re.match(PROPBANK_PATTERN, label) is not None


def get_best_qnode_for_mention_text(
    mention_or_text: Union[str, Mention],
    claim: Claim,
    amr: AMR,
    alignments: List[AMR_Alignment],
    spacy_model: Language,
    linking_model: WikidataLinkingClassifier,
    wn_lemmatizer: WordNetLemmatizer,
    qnode_tables: QnodeTables,
    device: str,
) -> Optional[WikidataQnode]:
    """Return the best WikidataQnode for a string within the claim sentence.

    First, if the string comes from a propbank frame, try a DWD lookup.
    Otherwise, run KGTK.
    """
    mention_text = None
    mention_id = None
    mention_span = None
    if isinstance(mention_or_text, str):
        mention_text = mention_or_text
        print("We're about to get the mention span...")
        mention_span = claim.get_offsets_for_text(mention_text, spacy_model.tokenizer)
    elif isinstance(mention_or_text, Mention):
        mention_text = mention_or_text.text
        mention_id = mention_or_text.mention_id
        mention_span = mention_or_text.span
    if not mention_text:
        return None

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
        if claim_variable_tokens and claim_variable_tokens[0] in token_list_for_node:
            variable_node_label = amr.nodes[node].strip('"')

    if not variable_node_label:
        logging.warning(
            "DWD lookup: could not find AMR node corresponding with XVariable/Claimer '%s'",
            mention_text,
        )
        node_label_is_pb = False
    else:
        node_label_is_pb = is_pb_label(variable_node_label)

    if variable_node_label and node_label_is_pb:
        best_qnodes = determine_best_qnodes(
            [variable_node_label],
            qnode_tables,
            spacy_model,
            linking_model,
            claim.claim_text,
            check_mappings_only=True,
        )
        if best_qnodes:
            best_qnode = best_qnodes[0]  # there should only be one
            score = float(best_qnode["score"]) if best_qnode.get("score") else 1.0
            return WikidataQnode(
                text=best_qnode.get("name"),
                mention_id=mention_id,
                doc_id=claim.doc_id,
                span=mention_span,
                confidence=score,
                description=best_qnode.get("definition"),
                from_query=best_qnode.get("pb"),
                qnode_id=best_qnode.get("qnode"),
            )

    # If no Qnode was found, try the entity mapping
    query_list: List[str] = list(filter(None, [mention_text, *claim_variable_tokens]))
    wd_qnode_from_entity_table = get_xpo_entity_result(
        query_list, qnode_tables.xpo_entities, wn_lemmatizer, claim.doc_id, mention_id, mention_span
    )
    if wd_qnode_from_entity_table:
        return wd_qnode_from_entity_table

    # Finally, try KGTK
    for query in query_list:
        claim_variable_links = disambiguate_refvar_kgtk(
            query, linking_model, claim.claim_sentence, k=1, device=device
        )
        claim_event_links = disambiguate_verb_kgtk(
            query, linking_model, claim.claim_text, k=1, device=device
        )
        # Combine the results
        all_claim_links = claim_variable_links
        all_claim_links["options"].extend(claim_event_links["options"])
        all_claim_links["other_options"].extend(claim_event_links["other_options"])
        all_claim_links["all_options"].extend(claim_event_links["all_options"])
        top_link = create_wikidata_qnodes(all_claim_links, claim, mention_id, mention_span)
        if top_link:
            return top_link
    return None


def get_wikidata_for_labeled_args(
    amr: AMR,
    alignments: List[AMR_Alignment],
    claim: Claim,
    args: MutableMapping[str, Any],
    spacy_model: Language,
    linking_model: WikidataLinkingClassifier,
    wn_lemmatizer: WordNetLemmatizer,
    qnode_tables: QnodeTables,
    device: str = CPU,
) -> MutableMapping[str, Any]:
    """Get WikiData for labeled arguments."""
    args_to_qnodes = {}
    for role, arg in args.items():
        qnode_selection = get_best_qnode_for_mention_text(
            arg,
            claim,
            amr,
            alignments,
            spacy_model,
            linking_model,
            wn_lemmatizer,
            qnode_tables,
            device,
        )
        if qnode_selection:
            args_to_qnodes[role] = ClaimArg(
                text=qnode_selection.text,
                mention_id=qnode_selection.mention_id,
                doc_id=qnode_selection.doc_id,
                span=qnode_selection.span,
                qnode_id=qnode_selection.qnode_id,
                description=qnode_selection.description,
                from_query=qnode_selection.from_query,
            )
    return args_to_qnodes


def get_claim_semantics(
    amr_sentence: AMR,
    amr_alignments: List[AMR_Alignment],
    claim: Claim,
    spacy_model: Language,
    linking_model: WikidataLinkingClassifier,
    wordnet_lemmatizer: WordNetLemmatizer,
    qnode_mappings: QnodeTables,
    device: str = CPU,
) -> List[ClaimSemantics]:
    """Disambiguate AMR sentence according to DWD overlay.

    Return all event qnodes and their corresponding arguments for the given claim sentence.
    """
    all_claim_semantics = []

    # Gather propbank nodes from resulting graph
    label_list_all = amr_sentence.get_ordered_node_labels()
    pb_label_list = [label for label in label_list_all if re.match(PROPBANK_PATTERN, label)]
    if not pb_label_list:
        logging.warning("No PropBank labels in the graph!")
        return []

    best_event_qnodes = determine_best_qnodes(
        pb_label_list,
        qnode_mappings,
        spacy_model,
        linking_model,
        claim.claim_sentence,
        device,
    )

    for event_qnode in best_event_qnodes:
        # Reverse search for the pb's source text
        # in order to get the associated char offsets
        event_span = None
        if event_qnode.get("pb"):
            pb_amr_node = get_node_from_pb(amr_sentence, event_qnode["pb"])
            event_tokens = amr_sentence.get_tokens_from_node(pb_amr_node, amr_alignments)
            if "<ROOT>" in event_tokens:
                event_tokens.remove("<ROOT>")
            event_text = remove_preceding_trailing_stop_words(" ".join(event_tokens))
            if not event_text:
                continue
            event_span = claim.get_offsets_for_text(event_text, spacy_model.tokenizer)
        else:
            pb_amr_node = ""
            logging.warning("No propbank label assigned to event qnode data: %s", event_qnode)

        wd: MutableMapping[str, Any] = {}
        if pb_amr_node and event_qnode.get("args"):
            labeled_args = get_all_labeled_args(
                amr_sentence, amr_alignments, pb_amr_node, event_qnode["args"]
            )
            wd = get_wikidata_for_labeled_args(
                amr_sentence,
                amr_alignments,
                claim,
                labeled_args,
                spacy_model,
                linking_model,
                wordnet_lemmatizer,
                qnode_mappings,
                device,
            )

        if event_qnode.get("score"):
            float_score = float(event_qnode["score"])
        else:
            float_score = 1.0

        claim_event = ClaimEvent(
            text=event_qnode.get("name"),
            doc_id=claim.doc_id,
            span=event_span,
            description=event_qnode.get("definition"),
            from_query=event_qnode.get("pb"),
            qnode_id=event_qnode.get("qnode"),
            confidence=float_score,
        )

        claim_args = {k: {"type": w} for k, w in wd.items()}
        all_claim_semantics.append(ClaimSemantics(event=claim_event, args=claim_args))

    return all_claim_semantics


def get_arg_roles_for_kgkt_event(
    event_qnode: str, original_xpo_table: Dict[str, Any]
) -> Optional[Dict[Any, Any]]:
    """If an event is chosen from kgtk, use the XPO table to potentially find the arguments."""
    events = original_xpo_table["events"]
    qnode_dict = events.get(f"DWD_{event_qnode}")
    if qnode_dict:
        arguments = create_arg_dict(qnode_dict)
        if arguments:
            return arguments
    return None


def determine_best_qnodes(
    pb_label_list: List[str],
    qnode_tables: QnodeTables,
    spacy_model: Language,
    linking_model: WikidataLinkingClassifier,
    sentence_text: str,
    device: str = CPU,
    check_mappings_only: bool = False,
) -> List[Dict[str, Any]]:
    """Return a list of qnode results from the pb-to-qnode mappings.

    The list will contain the best qnode result for each PropBank frame
    in the claim's AMR graph.

    We prioritize overlay results since they tend to be more precise.
    Otherwise, we also check the master table, and finally do a KGTK lookup.
    """
    chosen_event_qnodes = []
    for pb in pb_label_list:
        top_results_for_pb = []
        # Get result from the XPO table
        best_overlay_qnode = get_xpo_event_result(pb, qnode_tables.xpo_events, spacy_model)
        if best_overlay_qnode:
            top_results_for_pb.append(best_overlay_qnode)

        # Get result from the master table
        best_master_qnode = get_master_result(
            pb, qnode_tables.original_master, qnode_tables.pb_to_qnodes_master, spacy_model
        )
        if best_master_qnode:
            top_results_for_pb.append(best_master_qnode)

        if not check_mappings_only:
            # Finally, run a KGTK lookup
            best_kgtk_qnode = get_kgtk_result_for_event(pb, linking_model, sentence_text, device)
            if best_kgtk_qnode:
                arguments_from_table = get_arg_roles_for_kgkt_event(
                    best_kgtk_qnode["qnode"], qnode_tables.original_overlay
                )
                if arguments_from_table:
                    best_kgtk_qnode["args"] = arguments_from_table
                top_results_for_pb.append(best_kgtk_qnode)

        if top_results_for_pb:
            top_qnode, top_score = get_best_qnode_by_semantic_similarity(
                pb, top_results_for_pb, spacy_model
            )
            if top_qnode:
                if not top_qnode.get("score"):
                    top_qnode["score"] = top_score
                chosen_event_qnodes.append(top_qnode)

    if not chosen_event_qnodes:
        logging.warning("Couldn't find a qnode for pbs %s", pb_label_list)

    return chosen_event_qnodes


def create_wikidata_qnodes(
    link: Any,
    claim: Claim,
    mention_id: Optional[str] = None,
    mention_span: Optional[Tuple[int, int]] = None,
) -> Optional[WikidataQnode]:
    """Create WikiData Qnodes from links."""
    options = link["options"]
    other_options = link["other_options"]
    all_options = link["all_options"]
    best_option = None
    top_score = 0.0
    text = None
    qnode = None
    description = None
    score = 1.0
    if len(options) > 0:
        # For options and other_options, grab the first in the list
        best_option = options[0]
    elif len(other_options) > 0:
        best_option = other_options[0]

    if best_option:
        text = best_option["rawName"] if best_option["rawName"] else None
        qnode = best_option.get("qnode")
        description = best_option["definition"] if best_option["definition"] else None
        score = 1.0
    elif len(all_options) > 0:
        for option in all_options:
            score = option.get("linking_score", 0.0)
            if score > top_score:
                top_score = score
                best_option = option
        if best_option:
            text = best_option["label"][0] if best_option["label"] else None
            score = top_score
            qnode = best_option.get("qnode")
            description = best_option["description"][0] if best_option["description"] else None
    if best_option:
        return WikidataQnode(
            text=text,
            mention_id=mention_id,
            doc_id=claim.doc_id,
            span=mention_span,
            qnode_id=qnode,
            description=description,
            from_query=link["query"],
            confidence=score or 1.0,
        )

    logging.warning("No WikiData links found for '%s'.", link["query"])
    return None


def get_xpo_event_result(
    propbank_label: str,
    pb_mapping: Dict[str, Any],
    spacy_model: Language,
) -> Dict[str, Any]:
    """Returns the 'best' qnode from the overlay mapping.

    Given the set of PropBank labels along with whether the qnode came from the root label.
    """
    appended_qnode_data = {"pb": propbank_label}
    qnode_dicts = pb_mapping.get(propbank_label)
    if qnode_dicts:
        if len(qnode_dicts) == 1:
            appended_qnode_data.update(qnode_dicts[0])
            return appended_qnode_data
        # If there is more than one, do string similarity
        # (It's unlikely that there will be a ton)
        best_qnode, score = get_best_qnode_by_semantic_similarity(
            propbank_label, qnode_dicts, spacy_model
        )
        if best_qnode:
            appended_qnode_data["score"] = score  # type: ignore
            appended_qnode_data.update(best_qnode)
    if len(appended_qnode_data) > 1:
        return appended_qnode_data
    return {}


def get_master_result(
    propbank_label: str,
    original_master_table: Dict[str, Any],
    pb_mapping: MutableMapping[str, Any],
    spacy_model: Language,
) -> Dict[str, Any]:
    """Get Qnode from master JSON."""
    appended_qnode_data = {"pb": propbank_label}
    qnode_dicts = pb_mapping.get(propbank_label)
    if qnode_dicts:
        # Return this if there is only one result
        if len(qnode_dicts) == 1:
            appended_qnode_data.update(qnode_dicts[0])
            return appended_qnode_data
        # Else, try to find the "best" result
        # First find the qnode with the best string similarity
        # (to be used as backup)
        qnode_with_best_name, score = get_best_qnode_by_semantic_similarity(
            propbank_label, qnode_dicts, spacy_model
        )
        # Next, try to find the most general qnode
        most_general_qnode = get_most_general_qnode(
            propbank_label, qnode_dicts, original_master_table
        )
        if most_general_qnode:
            appended_qnode_data.update(most_general_qnode)
            return appended_qnode_data
        elif qnode_with_best_name:
            appended_qnode_data["score"] = score  # type: ignore
            appended_qnode_data.update(qnode_with_best_name)
            return appended_qnode_data

    return {}


def get_xpo_entity_result(
    potential_entity_strings: List[str],
    qnode_mapping: Dict[str, Any],
    lemmatizer: WordNetLemmatizer,
    doc_id: str,
    mention_id: Optional[str],
    mention_span: Optional[Tuple[int, int]],
) -> Optional[WikidataQnode]:
    """Look up each string in the entity mapping until a match is found."""
    for potential_entity in potential_entity_strings:
        potential_lemma = lemmatizer.lemmatize(potential_entity.lower())
        qnode_info = qnode_mapping.get(potential_lemma) or qnode_mapping.get(potential_entity)
        if qnode_info:
            return WikidataQnode(
                text=qnode_info["name"],
                mention_id=mention_id,
                doc_id=doc_id,
                span=mention_span,
                qnode_id=qnode_info["qnode"],
                description=qnode_info["definition"],
                from_query=potential_entity,
                confidence=1.0,
            )
    return None


def get_kgtk_result_for_event(
    propbank_label: str,
    linking_model: WikidataLinkingClassifier,
    claim_text: str,
    device: str = CPU,
) -> Dict[str, Any]:
    """Get the KGTK result for an event in the claim sentence."""
    formatted_pb = propbank_label.rsplit("-", 1)[0]
    qnode_info = disambiguate_verb_kgtk(formatted_pb, linking_model, claim_text, k=1, device=device)
    if qnode_info["options"]:
        selected_qnode = qnode_info["options"][0]
    elif qnode_info["other_options"]:
        selected_qnode = qnode_info["other_options"][0]
    elif qnode_info["all_options"]:
        selected_qnode = qnode_info["all_options"][0]  # Just selecting the first result
    else:
        selected_qnode = None

    if selected_qnode:
        definition = selected_qnode.get("definition")
        score = selected_qnode.get("score")
        if not definition and selected_qnode.get("description"):
            definition = selected_qnode["description"][0]
        if selected_qnode.get("rawName") or selected_qnode.get("label"):
            return {
                "pb": propbank_label,
                "name": selected_qnode.get("rawName") or selected_qnode["label"][0],
                "qnode": selected_qnode["qnode"],
                "definition": definition,
                "score": score,
            }
    return {}


def get_best_qnode_by_semantic_similarity(
    pb: str, qnode_dicts: List[Dict[str, str]], spacy_model: Language
) -> Tuple[Optional[Dict[str, Any]], float]:
    """Return the best qnode name for the given PropBank frame ID.

    This computes semantic similarity between two strings by
    comparing their word vectors from the spaCy model.
    """
    logging.info("Getting similarity scores between %s and its qnodes...", pb)
    best_score = 0.0
    # Load qnode names
    qnames_to_dicts = {qdict["name"]: qdict for qdict in qnode_dicts}
    qnode_names = list(qnames_to_dicts.keys())
    best_qname = qnode_names[0]  # Use as default to start
    # PB example: test-01 -- get text before sense number
    pb_string = pb.rsplit("-", 1)[0].replace("-", " ")
    pb_doc = spacy_model(pb_string)
    for qnode_name in qnode_names:
        formatted_qname = qnode_name.replace("_", " ")
        qnode_doc = spacy_model(formatted_qname)
        similarity_score = pb_doc.similarity(qnode_doc)
        if similarity_score > best_score:
            best_score = similarity_score
            best_qname = qnode_name
    if best_score == 0.0:
        logging.warning("No best score found for %s.", pb)
    return qnames_to_dicts.get(best_qname), best_score


def get_most_general_qnode(
    pb: str, qnode_dicts: List[Dict[str, str]], original_master_table: Any
) -> Optional[Dict[str, str]]:
    """Get the most general qnode given a propbank verb."""
    ids_to_qdicts: Dict[str, Dict[str, str]] = {
        qnode_dict["qnode"]: qnode_dict for qnode_dict in qnode_dicts
    }
    qnode_ids: Set[str] = {qid for qid, _ in ids_to_qdicts.items()}
    qnode_options_map: Dict[str, Dict[str, List[str]]] = {}
    for og_qdict in original_master_table:
        current_qnode = og_qdict.get("id")
        if current_qnode in qnode_ids:
            parents = [parent.rsplit("_", 1)[-1] for parent in og_qdict["parents"]]
            qnode_options_map[current_qnode] = {
                "name": og_qdict["name"].split("_Q")[0],
                "parent_nodes": parents,
            }

    def narrow_down_qnode(qnode_list: List[str]) -> Optional[str]:
        keep_list = []
        for qnode in qnode_list:
            if ids_to_qdicts[qnode]["name"] == pb:
                return qnode
            for qparent in qnode_options_map[qnode]["parent_nodes"]:
                if qparent in qnode_ids:
                    keep_list.append(qparent)
        if keep_list:
            if len(keep_list) == 1:
                return str(keep_list[0])
            else:
                return narrow_down_qnode(keep_list)
        return None

    most_general_node = narrow_down_qnode(list(qnode_ids))
    if most_general_node:
        return ids_to_qdicts.get(most_general_node)
    return None
