"""You will need to run this in the transition-amr-parser env."""
from collections import defaultdict
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, MutableMapping, Optional, Set, Tuple

from amr_utils.alignments import AMR_Alignment
from amr_utils.amr import AMR  # pylint: disable=import-error
from nltk.corpus import stopwords
from spacy.language import Language

from cdse_covid.claim_detection.claim import Claim
from cdse_covid.semantic_extraction.mentions import ClaimArg, ClaimEvent, ClaimSemantics
from cdse_covid.semantic_extraction.utils.amr_extraction_utils import (
    PROPBANK_PATTERN,
    create_node_to_token_dict,
)
from wikidata_linker.wikidata_linking import disambiguate_kgtk

OVERLAY = "overlay"
MASTER = "master"

PARENT_DIR = Path(__file__).parent
ORIGINAL_MASTER_TABLE = PARENT_DIR / "resources" / "qe_master.json"

STOP_WORDS = set(stopwords.words("english")).union({"like"})


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
                    role = qnode_args[framenet_arg]["text_role"]
                    node_label = amr.nodes[arg_node]
                    token_of_node = node_labels_to_tokens.get(arg_node)
                    if not token_of_node:
                        labeled_args[role] = node_label.rsplit("-", 1)[0]
                    elif any(token_of_node.endswith(f" {stop_word}") for stop_word in STOP_WORDS):
                        # Sometimes the extracted token or part of it is irrelevant
                        labeled_args[role] = token_of_node.rsplit(" ")[0]
                    else:
                        labeled_args[role] = token_of_node
    return labeled_args


def get_wikidata_for_labeled_args(
    amr: AMR, claim: Claim, args: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Get WikiData for labeled arguments."""
    args_to_qnodes = {}
    for role, arg in args.items():
        qnode_info = disambiguate_kgtk(" ".join(amr.tokens), arg, no_ss_model=True, k=1)
        if qnode_info["options"]:
            qnode_selection = qnode_info["options"][0]
        elif qnode_info["all_options"]:
            qnode_selection = qnode_info["all_options"][0]
        else:
            qnode_selection = None

        claim_arg = None
        if qnode_selection:
            claim_arg = ClaimArg(
                text=qnode_selection.get("rawName"),
                doc_id=claim.doc_id,
                span=claim.get_offsets_for_text(arg),
                qnode_id=qnode_selection["qnode"],
                description=qnode_selection.get("definition"),
                from_query=arg,
            )

        args_to_qnodes[role] = claim_arg
    return args_to_qnodes


def load_tables() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load propbank-to-qnode mappings from files.

    If they aren't found where expected, they will be generated
    from their respective original mapping files.
    """
    master_table_path = PARENT_DIR / "resources" / "pb_to_qnode_master.json"
    if not master_table_path.exists():
        logging.info("Could not find `pb_to_qnode_master.json`; generating it now")
        generate_master_dict(ORIGINAL_MASTER_TABLE, master_table_path)
    overlay_table_path = PARENT_DIR / "resources" / "pb_to_qnode_overlay.json"
    if not overlay_table_path.exists():
        logging.info("Could not find `pb_to_qnode_overlay.json`; generating it now")
        generate_overlay_dict(PARENT_DIR / "resources" / "xpo_v3.2_freeze.json", overlay_table_path)
    # Load both qnode mappings
    with open(master_table_path, "r", encoding="utf-8") as in_json:
        pbs_to_qnodes_master = json.load(in_json)
    with open(overlay_table_path, "r", encoding="utf-8") as in_json_2:
        pbs_to_qnodes_overlay = json.load(in_json_2)

    return pbs_to_qnodes_master, pbs_to_qnodes_overlay


def get_claim_semantics(
    amr_sentence: AMR, amr_alignments: List[AMR_Alignment], claim: Claim, spacy_model: Language
) -> Optional[ClaimSemantics]:
    """Disambiguate AMR sentence according to DWD overlay."""
    # Make both tables
    pbs_to_qnodes_master, pbs_to_qnodes_overlay = load_tables()

    # Gather propbank nodes from resulting graph
    label_list_all = amr_sentence.get_ordered_node_labels()
    pb_label_list = [label for label in label_list_all if re.match(PROPBANK_PATTERN, label)]
    if not pb_label_list:
        logging.warning("No PropBank labels in the graph!")
        return None

    best_qnode = determine_best_qnode(
        pb_label_list, pbs_to_qnodes_overlay, pbs_to_qnodes_master, amr_sentence, spacy_model
    )

    wd: MutableMapping[str, Any] = {}
    if best_qnode and best_qnode.get("args"):
        node = get_node_from_pb(amr_sentence, best_qnode["pb"])
        labeled_args = get_all_labeled_args(amr_sentence, amr_alignments, node, best_qnode["args"])
        wd = get_wikidata_for_labeled_args(amr_sentence, claim, labeled_args)

    claim_event = ClaimEvent(
        text=best_qnode.get("name"),
        doc_id=claim.doc_id,
        description=best_qnode.get("definition"),
        from_query=best_qnode.get("pb"),
        qnode_id=best_qnode.get("qnode"),
    )
    claim_args = {k: {"identity": w} for k, w in wd.items()}
    return ClaimSemantics(event=claim_event, args=claim_args)


def determine_best_qnode(
    pb_label_list: List[str],
    pbs_to_qnodes_overlay: Dict[str, Any],
    pbs_to_qnodes_master: Dict[str, Any],
    amr: AMR,
    spacy_model: Language,
    check_mappings_only: bool = False,
) -> Dict[str, Any]:
    """Return list of qnode results from the overlay.

    We prioritize overlay results since they tend to be more precise.
    If the result is from the root verb, we automatically go with that.
    Otherwise, we also check the master table, and finally do a KGTK lookup.
    """
    ranked_qnodes = []
    root_label = amr.nodes[amr.root]
    best_overlay_qnode, overlay_result_from_root = get_overlay_result(
        pb_label_list, pbs_to_qnodes_overlay, root_label, spacy_model
    )

    if best_overlay_qnode:
        if overlay_result_from_root:
            return best_overlay_qnode
        else:
            ranked_qnodes.append(best_overlay_qnode)

    # Get result from the master table
    best_master_qnode, master_result_from_root = get_master_result(
        pb_label_list, pbs_to_qnodes_master, ORIGINAL_MASTER_TABLE, root_label, spacy_model
    )

    # Prioritize the master result if it is from the root node
    # or there is no result from the overlay.
    if master_result_from_root or not best_overlay_qnode:
        return best_master_qnode
    elif best_master_qnode and not master_result_from_root:
        ranked_qnodes.append(best_master_qnode)

    if check_mappings_only:
        logging.warning("Using node other than root to get event Qnode.")
        if ranked_qnodes:
            return ranked_qnodes[0]
    else:
        # Finally, run a KGTK lookup
        best_kgtk_qnode, kgtk_result_from_root = get_kgtk_result_for_event(pb_label_list, amr)
        # Prioritize the root if there is one
        if kgtk_result_from_root or not ranked_qnodes:
            return best_kgtk_qnode
        # Next, prioritize any other result we found in the tables
        logging.warning("Using node other than root to get event Qnode.")
        if ranked_qnodes:
            return ranked_qnodes[0]
        return best_kgtk_qnode
    return {}


def get_overlay_result(
    pb_label_list: List[str], pb_mapping: MutableMapping[str, Any], root_label: str, spacy_model: Language
) -> Tuple[Dict[str, Any], bool]:
    """Returns the 'best' qnode from the overlay mapping.

    Given the set of PropBank labels along with whether the qnode came from the root label.
    """
    for label in pb_label_list:
        is_root = label == root_label
        appended_qnode_data = {"pb": label}
        qnode_dicts = pb_mapping.get(label)
        if qnode_dicts:
            if len(qnode_dicts) == 1:
                appended_qnode_data.update(qnode_dicts[0])
                return appended_qnode_data, is_root
            # If there is more than one, do string similarity
            # (It's unlikely that there will be a ton)
            best_qnode = get_best_qnode_by_semantic_similarity(label, qnode_dicts, spacy_model)
            if best_qnode:
                appended_qnode_data.update(best_qnode)
        if len(appended_qnode_data) > 1:
            return appended_qnode_data, is_root
    return {}, False


def get_master_result(
    pb_label_list: List[str],
    pb_mapping: MutableMapping[str, Any],
    original_mapping: Path,
    root_label: str,
    spacy_model: Language,
) -> Tuple[Dict[str, Any], bool]:
    """Get Qnode from master JSON."""
    for label in pb_label_list:
        is_root = label == root_label
        appended_qnode_data = {"pb": label}
        qnode_dicts = pb_mapping.get(label)
        if qnode_dicts:
            # Return this if there is only one result
            if len(qnode_dicts) == 1:
                appended_qnode_data.update(qnode_dicts[0])
                return appended_qnode_data, is_root
            # Else, try to find the "best" result
            # First find the qnode with the best string similarity
            # (to be used as backup)
            qnode_with_best_name = get_best_qnode_by_semantic_similarity(label, qnode_dicts, spacy_model)
            # Next, try to find the most general qnode
            most_general_qnode = get_most_general_qnode(label, qnode_dicts, original_mapping)
            if most_general_qnode:
                appended_qnode_data.update(most_general_qnode)
                return appended_qnode_data, is_root
            elif qnode_with_best_name:
                appended_qnode_data.update(qnode_with_best_name)
                return appended_qnode_data, is_root

    return {}, False


def get_kgtk_result_for_event(pb_label_list: List[str], amr: AMR) -> Tuple[Dict[str, Any], bool]:
    """Get the KGTK result for an event in the claim sentence."""
    for pb_label in pb_label_list:
        is_root = pb_label == amr.nodes[amr.root]
        formatted_pb = pb_label.rsplit("-", 1)[0]
        qnode_info = disambiguate_kgtk(" ".join(amr.tokens), formatted_pb, no_ss_model=True, k=1)
        if qnode_info["options"]:
            selected_qnode = qnode_info["options"][0]
        elif qnode_info["all_options"]:
            selected_qnode = qnode_info["all_options"][0]
        else:
            selected_qnode = None

        if selected_qnode:
            return {
                "pb": pb_label,
                "name": selected_qnode.get("rawName") or selected_qnode["label"][0],
                "qnode": selected_qnode["qnode"],
                "definition": selected_qnode.get("definition") or selected_qnode["description"][0],
            }, is_root
    return {}, False


def get_best_qnode_by_semantic_similarity(
        pb: str, qnode_dicts: List[Dict[str, str]], spacy_model: Language
) -> Optional[Dict[str, str]]:
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
    return qnames_to_dicts.get(best_qname)


def get_string_bigrams(string: str) -> Set[str]:
    """Given a string, return the set of bigrams.

    Example: "cure" --> {'cu', 'ur', 're'}
    """
    return {"".join(string[x : x + 2]) for x in range(len(string) - 1)}


def get_most_general_qnode(
    pb: str, qnode_dicts: List[Dict[str, str]], original_mapping_path: Path
) -> Optional[Dict[str, str]]:
    """Get the most general qnode given a propbank verb."""
    ids_to_qdicts: Dict[str, Dict[str, str]] = {
        qnode_dict["qnode"]: qnode_dict for qnode_dict in qnode_dicts
    }
    qnode_ids: Set[str] = {qid for qid, _ in ids_to_qdicts.items()}
    qnode_options_map: Dict[str, Dict[str, List[str]]] = {}
    with open(original_mapping_path, "r", encoding="utf-8") as og_in_json:
        original_master_table = json.load(og_in_json)
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


def generate_master_dict(master_table_path: Path, pb_master: Path) -> None:
    """Generate master dict between PropBank and DWD."""
    with open(master_table_path, "r", encoding="utf-8") as in_json:
        qnode_mapping = json.load(in_json)
    pbs_to_qnodes = defaultdict(list)
    for qnode_dict in qnode_mapping:
        qnode = qnode_dict.get("id")
        qname = qnode_dict.get("name").split("_Q")[0]
        qdescr = qnode_dict.get("def")

        args = qnode_dict.get("roles")
        final_args = {}
        for arg_role, constraints in args.items():

            formatted_constraints = []
            for constraint in constraints:
                constraint_string = constraint.pop()
                if constraint_string != "None":
                    string_split = constraint_string.split("_")
                    arg_qnode = string_split[-1]
                    arg_qname = "_".join(string_split[0 : len(string_split) - 1])
                    arg_qname = arg_qname.replace("+", "")  # Remove weird '+' sign
                    formatted_constraint = {"name": arg_qname, "wd_node": arg_qnode}
                    formatted_constraints.append(formatted_constraint)

            arg_parts = arg_role.split("-")
            final_args[arg_parts[0]] = {
                "constraints": formatted_constraints,
                "text_role": "-".join(arg_parts[1:]),
            }

        qnode_summary = {
            "qnode": qnode,
            "name": qname,
            "definition": qdescr,
            "args": final_args,
        }

        pb = str(qnode_dict["pb"]).replace(".", "-")
        pbs_to_qnodes[pb].append(qnode_summary)
    with open(pb_master, "w+", encoding="utf-8") as out_json:
        json.dump(pbs_to_qnodes, out_json, indent=4)


def generate_overlay_dict(overlay_path: Path, pb_overlay: Path) -> None:
    """Create a PB-to-Qnode dict from the DWD overlay."""
    with open(overlay_path, "r", encoding="utf-8") as in_json:
        qnode_mapping = json.load(in_json)["events"]

    pbs_to_qnodes = defaultdict(list)
    # Use this to keep track of which qnodes have been added
    # to the final mapping (there are several repeats)
    pbs_to_qnodes_only: Dict[str, Set[str]] = defaultdict(set)

    def add_qnode_to_mapping(pb_label: str, qnode: str, qnode_info: Dict[str, Any]) -> None:
        """Add qnode data to the pb-to-qnode mapping.

        Check first that a qnode's data hasn't been added to the list
        associated with the PB, then add it to the mapping.
        """
        pb = str(pb_label).replace(".", "-").replace("_", "-")
        if qnode not in pbs_to_qnodes_only[pb]:
            pbs_to_qnodes[pb].append(qnode_info)
            pbs_to_qnodes_only[pb].add(qnode)

    for _, data in qnode_mapping.items():
        qnode = data.get("wd_qnode")
        qname = data.get("name")
        qdescr = data.get("wd_description")
        pb_roleset = data.get("pb_roleset")

        args = data.get("arguments")
        final_args = {}
        for arg in args:
            arg_name = arg["name"]
            arg_parts = arg_name.split("_")
            arg_pos = arg_parts[1] if arg_parts[0] in ["AM", "Ax", "mnr"] else arg_parts[0]
            if arg_pos:
                final_args[arg_pos] = {
                    "constraints": arg["constraints"],
                    "text_role": "-".join(arg_parts[2:]),
                }

            qnode_summary = {
                "qnode": qnode,
                "name": qname,
                "definition": qdescr,
                "args": final_args,
            }
            if pb_roleset:
                add_qnode_to_mapping(pb_roleset, qnode, qnode_summary)

            # Add "other PB rolesets"
            ldc_types = data.get("ldc_types")
            if ldc_types:
                for ldc_type in ldc_types:
                    pb_list = ldc_type.get("other_pb_rolesets")
                    for pb_item in pb_list:
                        add_qnode_to_mapping(pb_item, qnode, qnode_summary)

    with open(pb_overlay, "w+", encoding="utf-8") as out_json:
        json.dump(pbs_to_qnodes, out_json, indent=4)
