"""Methods for generating and loading Qnode mappings."""
from collections import defaultdict
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, Set

PARENT_DIR = Path(__file__).parent

# Set of PropBank frames to ignore when extracting events from AMRs
# This isn't an exhaustive set, but this seems to cover the most common ones
# observed in the output.
UNINFORMATIVE_PB_FRAMES = {
    "be-02",
    "close-13",
    "general-02",
    "have-03",
    "have-concession-91",
    "have-manner-91",
    "have-mod-91",
    "have-mode-91",
    "have-name-91",
    "have-ord-91",
    "have-org-role-91",
    "have-part-91",
    "have-polarity-91",
    "have-rel-role-91",
    "have-subevent-91",
    "have-value-91",
    "look-01",
    "look-02",
    "look-04",
    "look-09",
    "same-01",
    "say-01",
    "see-01",
    "see-02",
    "see-03",
    "see-04",
    "see-05"
}

@dataclass
class QnodeTables:
    """A collection of mappings used to link PropBank frames and other strings to Qnodes."""

    original_master: Dict[Any, Any]
    original_overlay: Dict[str, Any]
    pb_to_qnodes_master: Dict[str, Any]
    xpo_events: Dict[str, Any]
    xpo_entities: Dict[str, Any]


def load_tables() -> QnodeTables:
    """Load qnode mappings from files.

    If they aren't found where expected, they will be generated
    from their respective original mapping files.
    """
    # Load master tables
    original_master_table_path = PARENT_DIR / "resources" / "qe_master.json"
    with open(original_master_table_path, "r", encoding="utf-8") as master:
        original_master_table = json.load(master)
    master_table_path = PARENT_DIR / "resources" / "pb_to_qnode_master.json"
    if not master_table_path.exists():
        logging.info("Could not find `pb_to_qnode_master.json`; generating it now")
        generate_master_dict(original_master_table, master_table_path)

    # Load overlay tables
    original_xpo_table_path = PARENT_DIR / "resources" / "xpo_v3.2_freeze.json"
    with open(original_xpo_table_path, "r", encoding="utf-8") as xpo:
        original_xpo_table = json.load(xpo)
    xpo_event_table_path = PARENT_DIR / "resources" / "pb_to_qnode_overlay.json"
    if not xpo_event_table_path.exists():
        logging.info("Could not find `pb_to_qnode_overlay.json`; generating it now")
        generate_xpo_event_dict(original_xpo_table, xpo_event_table_path)
    xpo_entity_table_path = PARENT_DIR / "resources" / "names_to_qnode_overlay.json"
    if not xpo_entity_table_path.exists():
        logging.info("Could not find `names_to_qnode_overlay.json`; generating it now")
        generate_xpo_entity_dict(original_xpo_table, xpo_entity_table_path)

    # Load the three qnode mappings
    with open(master_table_path, "r", encoding="utf-8") as master_json:
        pbs_to_qnodes_master = json.load(master_json)
    with open(xpo_event_table_path, "r", encoding="utf-8") as overlay_json:
        pbs_to_qnodes_overlay = json.load(overlay_json)
    with open(xpo_entity_table_path, "r", encoding="utf-8") as entity_json:
        names_to_qnodes = json.load(entity_json)

    return QnodeTables(
        original_master=original_master_table,
        original_overlay=original_xpo_table,
        pb_to_qnodes_master=pbs_to_qnodes_master,
        xpo_events=pbs_to_qnodes_overlay,
        xpo_entities=names_to_qnodes,
    )


def generate_master_dict(og_master: Dict[Any, Any], pb_master: Path) -> None:
    """Generate master dict between PropBank and DWD."""
    pbs_to_qnodes = defaultdict(list)
    for qnode_dict in og_master:
        # Don't add qnode info if there's no description
        qdescr = qnode_dict.get("def")
        if not qdescr:
            continue
        qnode = qnode_dict.get("id")
        qname = qnode_dict.get("name").split("_Q")[0]

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

        qnode_summary = {"qnode": qnode, "name": qname, "definition": qdescr, "args": final_args}

        pb = str(qnode_dict["pb"]).replace(".", "-")
        pbs_to_qnodes[pb].append(qnode_summary)
    with open(pb_master, "w+", encoding="utf-8") as out_json:
        json.dump(pbs_to_qnodes, out_json, indent=4)


def create_arg_dict(qnode_dict: Dict[str, Any]) -> Dict[Any, Any]:
    """Generate an argument mapping from the XPO table's qnode info."""
    args = qnode_dict.get("arguments")
    final_args = {}
    if args:
        for arg in args:
            arg_name = arg["name"]
            arg_parts = arg_name.split("_")
            arg_pos = arg_parts[1] if arg_parts[0] in ["AM", "Ax", "mnr"] else arg_parts[0]
            if arg_pos:
                final_args[arg_pos] = {
                    "constraints": arg["constraints"],
                    "text_role": "-".join(arg_parts[2:]),
                }
    return final_args


def generate_xpo_event_dict(og_xpo: Dict[str, Any], pb_xpo: Path) -> None:
    """Create a PB-to-Qnode dict from the DWD overlay."""
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

    event_data = og_xpo["events"]
    for _, data in event_data.items():
        qdescr = data.get("wd_description")
        if not qdescr:
            continue
        qnode = data.get("wd_qnode")
        qname = data.get("name")
        pb_roleset = data.get("pb_roleset")

        final_args = create_arg_dict(data)

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

    with open(pb_xpo, "w+", encoding="utf-8") as out_json:
        json.dump(pbs_to_qnodes, out_json, indent=4)


def generate_xpo_entity_dict(og_xpo: Dict[str, Any], entity_overlay: Path) -> None:
    """Create an entity-to-Qnode dict from the DWD overlay."""
    entities_to_qnodes: Dict[str, Dict[str, str]] = {}
    entities_to_qnodes_only: Dict[str, Set[str]] = defaultdict(set)

    def add_qnode_to_mapping(qnode: str, qnode_info: Dict[str, Any]) -> None:
        """Add qnode data to the pb-to-qnode mapping.

        Check first that a qnode's data hasn't been added to the list
        associated with the name, then add it to the mapping.
        """
        name = qnode_info["name"]
        if qnode not in entities_to_qnodes_only[name]:
            entities_to_qnodes[name] = qnode_info
            entities_to_qnodes_only[name].add(qnode)

    entity_data = og_xpo["entities"]
    for data in entity_data.values():
        qdescr = data.get("wd_description")
        if not qdescr:
            continue
        qnode = data.get("wd_qnode")
        qname = data.get("name")

        if qname:
            cleaned_qname = qname.replace("_", " ")
            qnode_summary = {
                "qnode": qnode,
                "name": cleaned_qname,
                "definition": qdescr,
            }
            add_qnode_to_mapping(qnode, qnode_summary)

    with open(entity_overlay, "w+", encoding="utf-8") as out_json:
        json.dump(entities_to_qnodes, out_json, indent=4)
