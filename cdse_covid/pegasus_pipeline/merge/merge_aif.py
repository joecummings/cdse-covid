"""Merge AIF documents."""
import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Set, Tuple

from rdflib import Graph, URIRef
from rdflib.query import ResultRow
from tqdm import tqdm
from vistautils.key_value import byte_key_value_source_from_params
from vistautils.parameters import Parameters

from cdse_covid.pegasus_pipeline.merge.aif_models import Span

JUSTIFICATIONS = [
    URIRef(
        "https://raw.githubusercontent.com/NextCenturyCorporation/AIDA-Interchange-Format/master/java/src/main/resources/com/ncc/aif/ontologies/InterchangeOntology#informativeJustification"
    ),
    URIRef(
        "https://raw.githubusercontent.com/NextCenturyCorporation/AIDA-Interchange-Format/master/java/src/main/resources/com/ncc/aif/ontologies/InterchangeOntology#justifiedBy"
    ),
    URIRef(
        "https://raw.githubusercontent.com/NextCenturyCorporation/AIDA-Interchange-Format/master/java/src/main/resources/com/ncc/aif/ontologies/InterchangeOntology#TextJustification"
    ),
]

SOURCE_DOC = URIRef(
    "https://raw.githubusercontent.com/NextCenturyCorporation/AIDA-Interchange-Format/master/java/src/main/resources/com/ncc/aif/ontologies/InterchangeOntology#sourceDocument"
)
PROTOTYPE = URIRef(
    "https://raw.githubusercontent.com/NextCenturyCorporation/AIDA-Interchange-Format/master/java/src/main/resources/com/ncc/aif/ontologies/InterchangeOntology#prototype"
)


def get_claim_semantics_for_claim(graph: Graph, claim: URIRef) -> Set[ResultRow]:
    """Get Claim Semantics for a given Claim."""
    return set(
        graph.query(
            """
        SELECT ?cs
        WHERE {
            ?claim aida:claimSemantics ?cs .
        }
        """,
            initBindings={"claim": claim},
        )
    )


def get_claims(graph: Graph) -> Set[ResultRow]:
    """Get all claims on document graph."""
    return set(graph.query("SELECT ?c WHERE { ?c a aida:Claim . }"))


def get_span_for_rdf_obj(
    graph: Graph, rdf_obj: Optional[URIRef] = None, *, provenance: Optional[str] = "ISI"
) -> Optional[Span]:
    """Get span for given RDF object."""
    if not rdf_obj:
        logging.warning("No rdf object to look up.")
        return None

    span_info = set(
        graph.query(
            """
            SELECT ?confidence ?end ?start ?source
            WHERE {
                ?obj aida:justifiedBy ?justifiedBy .

                ?justifiedBy aida:endOffsetInclusive ?end ;
                    aida:startOffset ?start ;
                    aida:source ?source .

                OPTIONAL {
                    ?justifiedBy aida:confidence ?confb .

                    ?confb aida:confidenceValue ?confidence .
                }
            }
            """,
            initBindings={"obj": rdf_obj},
        )
    )

    try:
        span_info_dict = span_info.pop().asdict()
        return Span(
            start=span_info_dict["start"].value,
            end=span_info_dict["end"].value,
            source=span_info_dict["source"],
            provenance=provenance,
        )
    except KeyError:
        logging.warning("No 'justifiedBy' element found for obj: %s", rdf_obj)
        return None


def load_from_key_value_store(kvs_path: Path) -> Mapping[str, Graph]:
    """Load ClaimDataset from a ZipKeyValueStore."""
    local_params = Parameters.from_mapping({"input": {"type": "zip", "path": str(kvs_path)}})

    graphs = {}
    with byte_key_value_source_from_params(local_params) as source:
        for file_name, aif_graph in tqdm(source.items()):
            graph = Graph()
            graph = graph.parse(source=aif_graph, format="turtle")
            graphs[file_name] = graph

    return graphs


def fuzzy_match(
    span: Tuple[int, int], keys: Iterable[Tuple[int, int]]
) -> Optional[Tuple[int, int]]:
    """Fuzzy match claim spans."""
    start = span[0]
    end = span[1]

    less_one_start = start - 1
    more_one_start = start + 1

    less_one_end = end - 1
    more_one_end = end + 1

    return next(
        (
            combo
            for combo in [
                (less_one_start, less_one_end),
                (less_one_start, end),
                (less_one_start, more_one_end),
                (start, less_one_start),
                (start, end),
                (start, more_one_end),
                (more_one_start, less_one_end),
                (more_one_start, end),
                (more_one_start, more_one_end),
            ]
            if combo in keys
        ),
        None,
    )


def link_ent(graph: Graph, uiuc_claim: URIRef, ent: URIRef) -> Graph:
    """Link entity to graph."""
    update_cs = """
        INSERT { ?claim aida:claimSemantics ?semantics . }
        WHERE { ?claim aida:claimSemantics ?s . }
    """

    graph.update(update_cs, initBindings={"claim": uiuc_claim, "semantics": ent})

    update_kes = """
        INSERT { ?claim aida:associatedKEs ?semantics . }
        WHERE { ?claim aida:associatedKEs ?s . }
    """

    graph.update(update_kes, initBindings={"claim": uiuc_claim, "semantics": ent})

    return graph


def create_ent(graph: Graph, ent: URIRef, props: Set[Tuple[URIRef, URIRef]]) -> Graph:
    """Create entity for graph."""
    query = """
    INSERT {
        ?ent ?p ?v
    }
    WHERE {
        OPTIONAL { ?ent ?p ?v }
    }
    """

    for prop, value in props:
        graph.update(query, initBindings={"ent": ent, "p": prop, "v": value})

    return graph


def create_cluster(
    graph: Graph, ent_as_cluster: URIRef, uiuc_ent: URIRef, props: Set[Tuple[URIRef, URIRef]]
) -> Graph:
    """Link the ISI cluster to the existing UIUC entity."""
    query = """
    INSERT {
        ?cluster ?p ?v
    }
    WHERE {
        OPTIONAL { ?cluster ?p ?v }
    }
    """

    for prop, value in props:
        graph.update(query, initBindings={"cluster": ent_as_cluster, "p": prop, "v": value})

    replacement_query = """
    DELETE {
        ?cluster aida:prototype ?v
    }
    INSERT {
        ?cluster aida:prototype ?v2
    }
    WHERE {
        OPTIONAL { ?cluster aida:prototype ?v }
    }
    """

    graph.update(
        replacement_query, initBindings={"cluster": ent_as_cluster, "p": PROTOTYPE, "v2": uiuc_ent}
    )

    return graph


def get_existing_ent_by_span(
    all_spans: MutableMapping[Tuple[int, int], URIRef],
    span: Optional[Span] = None,
    cluster_ent: Optional[URIRef] = None,
) -> Tuple[Optional[URIRef], Optional[URIRef]]:
    """Get an existing object in a graph identified by a span."""
    if not span:
        logging.warning(
            "No span for %s. This will be removed when all entities have a valid justification.",
            cluster_ent,
        )
        return None, None

    matching_span = fuzzy_match((span.start, span.end), all_spans.keys())
    matching_ent = None
    if matching_span:
        matching_ent = all_spans.get(matching_span)

    if matching_span and matching_ent:
        if (cluster_ent and "entity" in cluster_ent and "entities" in matching_ent) or (
            cluster_ent and "event" in cluster_ent and "events" in matching_ent
        ):
            return cluster_ent, matching_ent

    return None, None


def get_spans_for_all_ents(graph: Graph) -> MutableMapping[Tuple[int, int], URIRef]:
    """Get spans for all events and entities within a graph."""
    query = """
    SELECT ?ent ?start ?end
    WHERE {
        VALUES ?t { aida:Entity aida:Event }
        ?ent a ?t .
        ?ent aida:justifiedBy ?justifiedBy .

        ?justifiedBy aida:endOffsetInclusive ?end ;
            aida:startOffset ?start ;
            aida:source ?source .
    }
    """

    all_ents_with_spans = set(graph.query(query))

    all_spans = {}
    for ent in all_ents_with_spans:
        ent_dict = ent.asdict()
        span = Span(
            start=ent_dict["start"].value,
            end=ent_dict["end"].value,
            source="",
            provenance="",
        )

        all_spans[(span.start, span.end)] = ent_dict["ent"]

    return all_spans


def get_all_properties(graph: Graph, elem: URIRef) -> Set[Tuple[URIRef, URIRef]]:
    """Get all properties for a given URIRef."""
    query = """
    SELECT DISTINCT ?property ?value
    WHERE {
        ?elem ?property ?value .
        OPTIONAL { ?elem a ?property . }
    }
    """
    return set(graph.query(query, initBindings={"elem": elem}))


def get_all_event_types_and_arguments(graph: Graph) -> Set[ResultRow]:
    """Get all events and their arguments in a graph."""
    query = """
    SELECT DISTINCT ?et
    WHERE { ?et a rdf:Statement }
    """
    return set(graph.query(query))


def replace_ents_in_event_types_and_args(
    graph: Graph, elem: URIRef, entity_dict: Dict[URIRef, URIRef]
) -> Graph:
    """Update the entities in the graph for the given element."""
    entities_to_replace = []
    selection_query = """
    SELECT ?v
    WHERE { ?elem ?p ?v }
    """
    element_values = set(graph.query(selection_query, initBindings={"elem": elem}))
    for value in element_values:
        replacement_value = entity_dict.get(value[0])
        if replacement_value:
            entities_to_replace.append((value[0], replacement_value))

    replacement_query = """
    DELETE { ?elem ?p ?v }
    INSERT { ?elem ?p ?v2 }
    WHERE { ?elem ?p ?v }
    """
    for isi_ent, uiuc_ent in entities_to_replace:
        graph.update(replacement_query, initBindings={"elem": elem, "v": isi_ent, "v2": uiuc_ent})

    return graph


def is_in_updated_graph(graph1: Graph, graph2: Graph, elem: URIRef) -> bool:
    """Check if event or argument is in the updated graph."""
    query = """
    SELECT DISTINCT ?subject
    WHERE {
        ?elem rdf:subject ?subject .
    }
    """
    subject = set(graph1.query(query, initBindings={"elem": elem}))

    s1 = subject.pop()

    if s1:
        query2 = """
        SELECT DISTINCT ?e
        WHERE {
            ?e a aida:Event .
        }
        """

        return bool(graph2.query(query2, initBindings={"e": s1[0]}))

    return False


def get_abstract_source_id(graph: Graph) -> Optional[str]:
    """Get source id from any text justification element."""
    query = """
    SELECT ?source
    WHERE {
        ?e a aida:TextJustification .

        ?e aida:sourceDocument ?source .
    }
    """
    source_id = set(graph.query(query))
    if source_id:
        source_id_pair = tuple(source_id.pop())
        return str(source_id_pair[0].value) + ".ttl"

    return None


def get_source_id(graph: Graph, claim: URIRef) -> ResultRow:
    """Get source id from a claim."""
    query = """
    SELECT DISTINCT ?source
    WHERE {
        ?claim aida:justifiedBy ?jb .

        ?jb aida:sourceDocument ?source .
    }
    """
    res = set(graph.query(query, initBindings={"claim": claim}))
    return res.pop()  # type: ignore


def is_cluster_uri(uri: URIRef) -> bool:
    """Check if uri refers to a cluster."""
    return uri.find("clusters") != -1


def get_event_or_entity_from_cluster(uri: URIRef, graph: Graph) -> Optional[URIRef]:
    """Get direct event or entity from cluster object."""
    props = get_all_properties(graph, uri)
    return next(
        (y for _, y in props if y.find("events") != -1 or y.find("entities") != -1),
        None,
    )


def main(isi_store: Path, uiuc_store: Path, output: Path) -> None:
    """Entrypoint to Merge script."""
    isi_graphs = load_from_key_value_store(isi_store)
    uiuc_graphs = load_from_key_value_store(uiuc_store)

    total_replaced_entities = 0

    for graph_id, uiuc_graph in tqdm(uiuc_graphs.items()):

        source_id_string = get_abstract_source_id(uiuc_graph)

        potential_isi_graph = isi_graphs.get(graph_id)
        if potential_isi_graph:

            isi_claims = get_claims(potential_isi_graph)
            isi_claims_with_spans = {}
            for claim in isi_claims:
                span = get_span_for_rdf_obj(potential_isi_graph, claim[0], provenance="ISI")
                if span:
                    isi_claims_with_spans[(span.start, span.end)] = claim

            uiuc_claims = get_claims(uiuc_graph)
            uiuc_claims_with_spans = {}
            for claim in uiuc_claims:
                span = get_span_for_rdf_obj(uiuc_graph, claim[0], provenance="UIUC")
                if span:
                    uiuc_claims_with_spans[(span.start, span.end)] = claim

            # get spans for all objects in the graph
            all_entity_spans = get_spans_for_all_ents(uiuc_graph)

            matching_entities = {}

            for uiuc_claim_span, uiuc_claim in uiuc_claims_with_spans.items():
                source_id = get_source_id(uiuc_graph, uiuc_claim[0])

                match = fuzzy_match(uiuc_claim_span, isi_claims_with_spans.keys())
                if match:
                    isi_claim = isi_claims_with_spans[match]
                    # get all claim semantics URIs (events & arguments)
                    claim_semantics = get_claim_semantics_for_claim(
                        potential_isi_graph, isi_claim[0]
                    )

                    if claim_semantics:
                        for ent in claim_semantics:
                            ent_as_uri: URIRef = ent[0]  # convert from ResultRow to URI

                            specific_ent = get_event_or_entity_from_cluster(
                                ent_as_uri, potential_isi_graph
                            )

                            # check if justified is on entity, but link the cluster instead of entity directly
                            cluster_ent, existing_ent = get_existing_ent_by_span(
                                all_entity_spans,
                                span=get_span_for_rdf_obj(
                                    potential_isi_graph, specific_ent, provenance="ISI"
                                ),
                                cluster_ent=ent_as_uri,
                            )

                            if specific_ent and cluster_ent and existing_ent:
                                total_replaced_entities += 1
                                uiuc_graph = link_ent(uiuc_graph, uiuc_claim[0], cluster_ent)
                                all_props = get_all_properties(potential_isi_graph, cluster_ent)
                                uiuc_graph = create_cluster(
                                    uiuc_graph, cluster_ent, existing_ent, all_props
                                )
                                matching_entities[specific_ent] = existing_ent
                            else:
                                uiuc_graph = link_ent(uiuc_graph, uiuc_claim[0], ent_as_uri)
                                all_props = get_all_properties(potential_isi_graph, ent_as_uri)
                                uiuc_graph = create_ent(uiuc_graph, ent_as_uri, all_props)

                                stack = list(all_props)
                                # dfs through all the properties to update graph
                                while stack:
                                    curr_elem = stack.pop()
                                    curr_elem_value = curr_elem[1]
                                    all_props_of_curr = get_all_properties(
                                        potential_isi_graph, curr_elem_value
                                    )

                                    prop_type = curr_elem[0]

                                    if prop_type in JUSTIFICATIONS:
                                        all_props_of_curr.add((SOURCE_DOC, source_id[0]))

                                    if all_props_of_curr:
                                        uiuc_graph = create_ent(
                                            uiuc_graph, curr_elem_value, all_props_of_curr
                                        )
                                        stack.extend(all_props_of_curr)

            all_event_types_and_args = get_all_event_types_and_arguments(potential_isi_graph)
            for elem in all_event_types_and_args:
                if matching_entities:
                    potential_isi_graph = replace_ents_in_event_types_and_args(
                        potential_isi_graph, elem[0], matching_entities
                    )
                if is_in_updated_graph(potential_isi_graph, uiuc_graph, elem[0]):
                    all_props = get_all_properties(potential_isi_graph, elem[0])
                    if all_props:
                        uiuc_graph = create_ent(uiuc_graph, elem[0], all_props)

        if source_id_string:
            uiuc_graph.serialize(output / source_id_string, format="turtle")

    logging.warning("Number of entities replaced: %s", total_replaced_entities)
    logging.info("Serialized merged graphs to %s", output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--isi-store", type=Path)
    p.add_argument("--uiuc-store", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    main(args.isi_store, args.uiuc_store, args.output)
