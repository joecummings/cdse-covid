"""You will need to run this in the transition-amr-parser env"""
import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any, List, Mapping, Set, Dict, Optional, Tuple

from amr_utils.amr import AMR
from cdse_covid.semantic_extraction.amr_extraction_utils import LocalAMR, load_amr_from_text_file
from wikidata_linker.wikidata_linking import disambiguate_kgtk

OVERLAY = "overlay"
MASTER = "master"

PARENT_DIR = Path(__file__).parent

@dataclass
class AMRToQNode:
    doc_id: str
    sentence_idx: int
    event_qnode: Mapping[str, Any]
    args_qnodes: Mapping[str, Any]


def get_node_from_pb(amr, pb_label: str) -> str:
    for id, node in amr.nodes.items():
        if node == pb_label:
            return id


def get_edges_from_node(amr, node: str) -> str:
    return [e for e in amr.edges if e[0] == node]


def is_valid_arg_type(arg_type: str) -> bool:
    return "ARG" in arg_type or "time" in arg_type or "location" in arg_type or "direction" in arg_type


# Currently pushing ARG0-of to ARG0
pb_args_to_framenet_args = {
    "ARG0": "A0",
    "ARG0-of": "A0",
    "ARG1": "A1",
    "ARG1-of": "A1",
    "ARG2": "A2",
    "ARG2-of": "A2",
    "ARG3": "A3",
    "ARG3-of": "A3",
    "ARG4": "A4",
    "ARG4-of": "A4",
    "location": "loc",
    "location-of": "loc",
    "time": "time",
    "time-of": "time",
    "direction": "dir",
    "direction-of": "dir"
}


def get_all_labeled_args(amr, node: str, qnode_args: Mapping[str, Any]) -> List[str]:
    potential_args = get_edges_from_node(amr, node)
    labeled_args = {}

    for arg in potential_args:
        if is_valid_arg_type(arg[1]):
            formatted_arg_type = arg[1].replace(":", "")
            framenet_arg = pb_args_to_framenet_args[formatted_arg_type]
            if qnode_args.get(framenet_arg):
                role = qnode_args[framenet_arg]["text_role"]
                labeled_args[role] = amr.nodes[arg[2]]
    return labeled_args


def get_wikidata_from_labeled_args(amr, args: List[str]) -> Mapping[str, Any]:
    args_to_qnodes = {}
    for role, arg in args.items():
        if "-" not in arg:
            qnode_info = disambiguate_kgtk(amr.tokens, arg, no_ss_model=True, k=1)
            if qnode_info["options"]:
                args_to_qnodes[role] = qnode_info["options"][0]
            elif qnode_info["all_options"]:
                args_to_qnodes[role] = qnode_info["all_options"][0]
            else:
                args_to_qnodes[role] = None
    return args_to_qnodes


def disambiguate_with_amr_v2(doc_id: str, idx: int, amr_sentence: AMR) -> AMRToQNode:

    # Make both tables
    master_table_path = PARENT_DIR / "resources" / "pb_to_qnode_master.json"
    original_master_table = PARENT_DIR / "resources" / "qe_master.json"
    if not master_table_path.exists():
        print("Could not find `pb_to_qnode_master.json`; generating it now")
        generate_master_dict(original_master_table, master_table_path)
    overlay_table_path = PARENT_DIR / "resources" / "pb_to_qnode_overlay.json"
    if not overlay_table_path.exists():
        print("Could not find `pb_to_qnode_overlay.json`; generating it now")
        generate_overlay_dict(PARENT_DIR / "resources" / "xpo_dwd_overlay_v2.json", overlay_table_path)

    # Load both qnode mappings
    print("Loading qnode mappings...")
    with open(master_table_path, 'r') as in_json:
        pbs_to_qnodes_master = json.load(in_json)
    with open(overlay_table_path, 'r') as in_json_2:
        pbs_to_qnodes_overlay = json.load(in_json_2)

    # Gather propbank nodes from resulting graph
    frame_label_pattern = r"[a-z-]+-[0-9]{2}"  # e.g. have-name-91
    
    # We're only dealing with one sentence here
    label_list_all = get_node_labels(amr_sentence.amr_string())

    pb_label_list = [
        label
        for label in label_list_all
        if re.match(frame_label_pattern, label)
    ]

    if not pb_label_list:
        print("No PropBank labels in the graph!")
        return
    # The first one is the root label
    root_label = pb_label_list[0]

    # return list of qnode results from the overlay
    # We prioritize overlay results since they tend to be more precise.
    # If the result is from the root verb, we automatically go with that.
    # Otherwise, we also check the master table.
    best_qnode, overlay_result_from_root = get_overlay_result(
        pb_label_list, pbs_to_qnodes_overlay, root_label
    )
    print(f"Best qnode from overlay: {best_qnode}")                

    if not best_qnode or not overlay_result_from_root:
        print("^Not from root")
        # get_master_result...
        best_master_qnode, master_result_from_root = get_master_result(
            pb_label_list, pbs_to_qnodes_master, original_master_table, root_label
        )
        print(f"Best qnode from master: {best_master_qnode}")
        # Prioritize the master result if it is from the root node
        # or there is no result from the overlay.
        if master_result_from_root or not best_qnode:
            best_qnode = best_master_qnode

    if best_qnode and best_qnode.get("args"):
        node = get_node_from_pb(amr_sentence, best_qnode["pb"])
        labeled_args = get_all_labeled_args(amr_sentence, node, best_qnode["args"])
        wd = get_wikidata_from_labeled_args(amr_sentence, labeled_args)
    else:
        wd = {}

    # Declare the qnode result
    return AMRToQNode(
        doc_id=doc_id,
        sentence_idx=idx,
        event_qnode=best_qnode,
        args_qnodes=wd
    )


def get_node_labels(amr_str: str) -> List[str]:
    """Return a list of node labels from an AMR graph starting from the root"""
    # Create edge dict
    lines = amr_str.split("\n")
    node_dict = {}
    edge_dict = defaultdict(list)
    root = None
    for line in lines:
        tab_separated = line.split("\t")
        if tab_separated[0] == "# ::node":
            node_dict[tab_separated[1]] = tab_separated[2]
        if tab_separated[0] == "# ::edge":
            edge_dict[tab_separated[4]].append(tab_separated[5])
        elif tab_separated[0] == "# ::root":
            root = tab_separated[1]
    print(edge_dict)

    # Create ordered list of node labels starting from the root
    label_list = [node_dict[root]]
    processed_nodes = {root}

    def fill_list(parent_list: List[str]) -> None:
        print(parent_list)
        child_nodes: List[str] = []
        for parent in parent_list:
            children = edge_dict.get(parent)
            if children:
                child_nodes.extend(
                    [child for child in children if child not in processed_nodes]
                )
        # First add labels on this level
        for child in child_nodes:
            label_list.append(node_dict[child])
            processed_nodes.add(child)
        # Process the children of the next level
        if child_nodes:
            fill_list(child_nodes)

    fill_list([root])
    return label_list


def get_overlay_result(
        pb_label_list: List[str], pb_mapping, root_label: str
) -> Tuple[Dict[str, str], bool]:
    """
    Returns the 'best' qnode from the overlay mapping given the set of PropBank labels
    along with whether the qnode came from the root label.
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
            best_qnode = get_best_qnode_by_string_similarity(label, qnode_dicts)
            if best_qnode:
                appended_qnode_data.update(best_qnode)
        if len(appended_qnode_data) > 1:
            return appended_qnode_data, is_root
    return {}, False


def get_master_result(
        pb_label_list: List[str], pb_mapping, original_mapping: str, root_label: str
) -> Tuple[Dict[str, str], bool]:
    for label in pb_label_list:
        qnode_dicts = pb_mapping.get(label)
        print(f"qnode dicts for {label}:\n{qnode_dicts}")
    for label in pb_label_list:
        is_root = label == root_label
        appended_qnode_data = {"pb": label}
        qnode_dicts = pb_mapping.get(label)
        print(f"qnode dicts for {label}:\n{qnode_dicts}")
        if qnode_dicts:
            # Return this if there is only one result
            if len(qnode_dicts) == 1:
                appended_qnode_data.update(qnode_dicts[0])
                return appended_qnode_data, is_root
            # Else, try to find the "best" result
            # First find the qnode with the best string similarity
            # (to be used as backup)
            qnode_with_best_name = get_best_qnode_by_string_similarity(
                label, qnode_dicts
            )
            # Next, try to find the most general qnode
            most_general_qnode = get_most_general_qnode(
                label, qnode_dicts, original_mapping
            )
            if most_general_qnode:
                appended_qnode_data.update(most_general_qnode)
                return appended_qnode_data, is_root
            elif qnode_with_best_name:
                appended_qnode_data.update(qnode_with_best_name)
                return appended_qnode_data, is_root

    return {}, False


def get_best_qnode_by_string_similarity(
        pb: str, qnode_dicts: List[Dict[str, str]]
) -> Dict[str, str]:
    """
    Return the best qnode name for the given PropBank frame ID
    based on string similarity score (Dice Coefficient)

    This assumes that all PBs and qnode names are at least
    two characters long.
    """
    print(f"Getting similarity scores between {pb} and its qnodes...")
    best_score = 0
    # Load qnode names
    qnames_to_dicts = {qdict["name"]: qdict for qdict in qnode_dicts}
    qnode_names = [qname for qname, qdict in qnames_to_dicts.items()]
    best_qname = qnode_names[0]  # Use as default to start
    # PB example: test-01 -- get text before sense number
    pb_string = pb.rsplit("-", 1)[0]
    pb_string_bigrams = get_string_bigrams(pb_string)
    # {"qname": {<bigrams>}, ...}
    qname_string_bigram_sets: Dict[str, Set[str]] = {
        qname: get_string_bigrams(qname) for qname in qnode_names
    }
    for qname, qname_string_bigrams in qname_string_bigram_sets.items():
        print(f"Evaluating {qname}...")
        overlap = len(pb_string_bigrams & qname_string_bigrams)
        dice_coefficient = overlap * 2.0/(
                len(pb_string_bigrams) + len(qname_string_bigrams)
        )
        print(f"{qname} score: {dice_coefficient}")
        if dice_coefficient > best_score:
            best_qname = qname
            best_score = dice_coefficient
    if best_score == 0.0:
        # Try to find word form matches
        print("No best score")
    best_qnode = qnames_to_dicts.get(best_qname)
    print(f"Most string-similar qnode: {best_qnode}\n")
    return best_qnode


def get_string_bigrams(string: str) -> Set[str]:
    """
    Given a string, return the set of bigrams.
    Example: "cure" --> {'cu', 'ur', 're'}
    """
    return {"".join(string[x:x+2]) for x in range(len(string)-1)}


def get_most_general_qnode(
        pb: str, qnode_dicts: List[Dict[str, str]], original_mapping_path: str
) -> Optional[Dict[str, str]]:
    ids_to_qdicts: Dict[str, Dict[str, str]] = {
        qnode_dict["qnode"]: qnode_dict for qnode_dict in qnode_dicts
    }
    qnode_ids: Set[str] = {qid for qid, _ in ids_to_qdicts.items()}
    qnode_options_map: Dict[str, Dict[str, List[str, str]]] = {}
    with open(original_mapping_path, 'r') as og_in_json:
        original_master_table = json.load(og_in_json)
    for og_qdict in original_master_table:
        current_qnode = og_qdict.get("id")
        if current_qnode in qnode_ids:
            parents = [parent.rsplit("_", 1)[-1] for parent in og_qdict["parents"]]
            qnode_options_map[current_qnode] = {
                "name": og_qdict["name"].split("_Q")[0],
                "parent_nodes": parents
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
                return keep_list[0]
            else:
                return narrow_down_qnode(keep_list)
        return None

    most_general_node = narrow_down_qnode(list(qnode_ids))
    print(f"Most general qnode found: {ids_to_qdicts.get(most_general_node)}")
    return ids_to_qdicts.get(most_general_node)


def generate_master_dict(master_table_path: str, pb_master: str):
    with open(master_table_path, 'r') as in_json:
        qnode_mapping = json.load(in_json)
    pbs_to_qnodes = defaultdict(list)
    for qnode_dict in qnode_mapping:
        abridged_qdict = {
            "qnode": qnode_dict.get("id"),
            "name": qnode_dict.get("name").split("_Q")[0],
            "definition": qnode_dict.get("def")
        }
        pb = str(qnode_dict["pb"]).replace(".", "-")
        pbs_to_qnodes[pb].append(abridged_qdict)
    with open(pb_master, 'w+') as out_json:
        json.dump(pbs_to_qnodes, out_json, indent=4)


def generate_overlay_dict(overlay_path: str, pb_overlay: str):
    """
    Create a PB-to-Qnode dict from the DWD overlay
    and save it as a json file.
    """
    with open(overlay_path, 'r') as in_json:
        qnode_mapping = json.load(in_json)["events"]
    pbs_to_qnodes = defaultdict(list)
    for _, data_list in qnode_mapping.items():
        for data in data_list:
            qnode = data.get("wd_node")
            qname = data.get("name")
            qdescr = data.get("wd_description")

            args = data.get("arguments")
            final_args = {}
            for arg in args:
                arg_name = arg["name"]
                arg_parts = arg_name.split("_")
                arg_pos = arg_parts[1] if arg_parts[0] in ["AM", "Ax", "mnr"] else arg_parts[0]
                if arg_pos:
                    final_args[arg_pos] = {
                        'constraints': arg["constraints"],
                        'text_role': "-".join(arg_parts[2:]),
                    }

            qnode_summary = {
                "qnode": qnode,
                "name": qname, 
                "definition": qdescr, 
                "args": final_args
            }

            ldc_types = data.get("ldc_types")
            for ldc_type in ldc_types:
                pb_list = ldc_type.get("pb_rolesets")
                for pb_item in pb_list:
                    pb = str(pb_item).replace(".", "-").replace("_", "-")
                    pbs_to_qnodes[pb].append(qnode_summary)
    with open(pb_overlay, 'w+') as out_json:
        json.dump(pbs_to_qnodes, out_json, indent=4)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--amr-graphs", type=str, help="Sentence to get qnodes for")
    args = arg_parser.parse_args()
    disambiguate_with_amr_v2(args.amr_graphs)
