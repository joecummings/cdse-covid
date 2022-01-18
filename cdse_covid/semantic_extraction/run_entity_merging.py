"""Connect coref entities through spans."""
import argparse
import csv
from pathlib import Path
import pickle
from typing import MutableMapping, Optional, Tuple

from cdse_covid.claim_detection.run_claim_detection import ClaimDataset
from cdse_covid.pegasus_pipeline.ingesters.edl_output_ingester import (  # pylint: disable=unused-import
    EDLEntity,
    EDLMention,
)
from cdse_covid.semantic_extraction.mentions import Mention, WikidataQnode
from wikidata_linker.wikidata_linking import (
    KGTK_EVENT_CACHE,
    KGTK_REFVAR_CACHE,
    get_request_kgtk,
    make_cache_path,
)

type_mapping_to_qnode = {
    "COM": WikidataQnode(
        text="physical object",
        qnode_id="Q223557",
        description="singular aggregation of substance(s) such as matter or radiation, with overall properties such as mass, position or momentum",
    ),  # physical object
    "PER": WikidataQnode(
        text="human",
        qnode_id="Q5",
        description="common name of Homo sapiens, unique extant species of the genus Homo",
    ),  # human
    "LOC": WikidataQnode(
        text="geographic location",
        qnode_id="Q2221906",
        description="point or an area on the Earth's surface or elsewhere",
    ),  # geographic location
    "GPE": WikidataQnode(
        text="geopolitical group",
        qnode_id="Q52110228",
        description="group of independent or autonomous territories sharing a given set of traits",
    ),  # geopolitical group
    "ORG": WikidataQnode(
        text="organization",
        qnode_id="Q43229",
        description="social entity (not necessarily commercial) uniting people into a structured group managing shared means to meet some needs, or to pursue collective goals",
    ),  # organization
    "FAC": WikidataQnode(
        text="facility", qnode_id="Q13226383", description="place for doing something"
    ),  # facility
    "VEH": WikidataQnode(
        text="vehicle",
        qnode_id="Q42889",
        description="mobile machine that transports people, animals or cargo",
    ),  # vehicle
    "MHI": WikidataQnode(
        text="health problem",
        qnode_id="Q2057971",
        description="condition negatively affecting the health of an organism",
    ),  # health problem
    "WEA": WikidataQnode(
        text="weapon", qnode_id="Q728", description="tool used to inflict damage or harm"
    ),  # weapon
}


def _contains(span1: Tuple[int, int], span2: Tuple[int, int]) -> bool:
    """Determine if span1 contains span2."""
    return span1[0] <= span2[0] and span1[1] >= span2[1]


def find_knowledge_entity(
    store: MutableMapping[Tuple[int, int], EDLMention],
    target_span: Optional[Tuple[int, int]],
    include_contains: bool = False,
) -> Optional[EDLMention]:
    """Find knowledge entity associated with *target span*."""
    if target_span is None:
        return None

    # UIUC uses exclusive char offsets so we have to subtract 1
    target_span = (target_span[0], target_span[1] - 1)

    possible_ke = store.get(target_span)
    if possible_ke:
        return possible_ke

    if include_contains:
        for span, entity in store.items():
            if _contains(target_span, span):
                return entity

    return None


def create_wikidata_qnode_from_id(mention: Mention, qnode_id: str) -> Optional[WikidataQnode]:
    """Get the rest of the qnode data from KGTK."""
    event_cache_file = make_cache_path(KGTK_EVENT_CACHE, qnode_id)
    refvar_cache_file = make_cache_path(KGTK_REFVAR_CACHE, qnode_id)
    kgtk_json = get_request_kgtk(qnode_id, refvar_cache_file, filter_results=False)
    kgtk_json += get_request_kgtk(qnode_id, event_cache_file, filter_results=False)
    # We assume there will only be one result from a Qnode ID query, if any
    selected_qnode = None
    if kgtk_json:
        kgtk_result = kgtk_json[0]
        selected_qnode = {
            "qnode": kgtk_result["qnode"],
            "rawName": kgtk_result["label"][0],
            "definition": kgtk_result["description"][0] if kgtk_result["description"] else "",
        }

    if selected_qnode:
        return WikidataQnode(
            text=selected_qnode.get("rawName"),
            mention_id=mention.mention_id,
            entity=mention.entity,
            doc_id=mention.doc_id,
            span=mention.span,
            qnode_id=qnode_id,
            description=selected_qnode.get("definition"),
            from_query=qnode_id,
        )
    return None


def load_freebase_to_qnode_mapping(
    original_map_tsv: Path, mapping_file: Path
) -> MutableMapping[str, str]:
    """Load the freebase-to-qnode mapping file if it exists.

    Otherwise, generate the mapping from the qnode-freebase file
    and save that data.
    """
    freebase_to_qnodes: MutableMapping[str, str] = {}
    if mapping_file.exists():
        with open(mapping_file, "rb") as handle:
            freebase_to_qnodes = pickle.load(handle)
    else:
        with open(original_map_tsv, "r", encoding="utf-8") as in_map:
            reader = csv.reader(in_map, delimiter="\t")
            for line in reader:
                freebase_to_qnodes[line[1]] = line[0]
        # Save mapping for loading later
        with open(mapping_file, "wb+") as out_map:
            pickle.dump(freebase_to_qnodes, out_map)

    return freebase_to_qnodes


def main(
    edl: Path,
    claims: Path,
    output: Path,
    include_contains: bool,
) -> None:
    """Run entity linking over claims."""
    with open(edl, "rb") as handle:
        edl_store = pickle.load(handle)

    claim_ds = ClaimDataset.load_from_key_value_store(claims)

    for claim in claim_ds:
        doc_id = claim.doc_id.split(".")[0]
        all_kes = edl_store[doc_id]
        if claim.x_variable:
            x_variable_mention = find_knowledge_entity(
                all_kes, claim.x_variable.span, include_contains
            )
            if x_variable_mention:
                # The x-variable is an entity, so make its qnode the identity qnode.
                # The _type_qnode will be replaced later, so we don't bother removing it.
                claim.x_variable_identity_qnode = claim.x_variable_type_qnode
                claim.x_variable.entity = x_variable_mention.parent_entity
                entity_qnode = claim.x_variable.entity.ent_link
                entity_type_qnode = claim.x_variable.entity.type_link
                if entity_qnode:
                    claim.x_variable_identity_qnode = create_wikidata_qnode_from_id(
                        claim.x_variable, entity_qnode
                    )
                if entity_type_qnode:
                    claim.x_variable_type_qnode = create_wikidata_qnode_from_id(
                        claim.x_variable, entity_type_qnode
                    )
                else:
                    # Get a qnode corresponding to the entity type
                    base_type = x_variable_mention.parent_entity.ent_type.split(".")[0]
                    claim.x_variable_type_qnode = type_mapping_to_qnode.get(base_type)

        if claim.claimer:
            claimer_mention = find_knowledge_entity(all_kes, claim.claimer.span, include_contains)
            if claimer_mention:
                # The claimer is an entity, so make its qnode the identity qnode
                claim.claimer_identity_qnode = claim.claimer_type_qnode
                claim.claimer.entity = claimer_mention.parent_entity
                entity_qnode = claim.claimer.entity.ent_link
                entity_type_qnode = claim.claimer.entity.type_link
                if entity_qnode:
                    claim.claimer_identity_qnode = create_wikidata_qnode_from_id(
                        claim.claimer, entity_qnode
                    )
                if entity_type_qnode:
                    claim.claimer_type_qnode = create_wikidata_qnode_from_id(
                        claim.claimer, entity_type_qnode
                    )
                else:
                    # Get a qnode corresponding to the entity type
                    base_type = claimer_mention.parent_entity.ent_type.split(".")[0]
                    claim.claimer_type_qnode = type_mapping_to_qnode.get(base_type)

        if claim.claim_semantics:
            for _, arg in claim.claim_semantics.args.items():
                if arg:
                    type_arg = arg["type"]
                    if type_arg:
                        arg_mention = find_knowledge_entity(
                            all_kes, type_arg.span, include_contains
                        )
                        if arg_mention:
                            arg["identity"] = type_arg
                            arg["identity"].entity = arg_mention.parent_entity
                            entity_qnode_id = arg["identity"].entity.ent_link
                            entity_type_qnode_id = arg["identity"].entity.type_link
                            if entity_qnode_id:
                                arg_qnode = create_wikidata_qnode_from_id(
                                    arg["identity"], entity_qnode_id
                                )
                                arg["identity"] = arg_qnode
                            if entity_type_qnode_id:
                                arg_type_qnode = create_wikidata_qnode_from_id(
                                    arg["type"], entity_type_qnode_id
                                )
                                arg["type"] = arg_type_qnode
                            else:
                                base_type = arg_mention.parent_entity.ent_type.split(".")[0]
                                arg["type"] = type_mapping_to_qnode.get(base_type)

    claim_ds.save_to_key_value_store(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--edl", help="option help", type=Path)
    p.add_argument("--claims", help="Claim information", type=Path)
    p.add_argument("--output", help="Path to output file.", type=Path)
    p.add_argument(
        "--include-contains",
        action="store_true",
        help="Include matches where span1 contains span2.",
    )
    args = p.parse_args()
    main(
        args.edl,
        args.claims,
        args.output,
        args.include_contains,
    )
