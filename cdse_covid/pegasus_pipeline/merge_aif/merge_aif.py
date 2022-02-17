"""Merge AIF documents."""
import argparse
import logging
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Set, Tuple

from rdflib import Graph, URIRef
from tqdm import tqdm
from vistautils.key_value import byte_key_value_source_from_params
from vistautils.parameters import Parameters

from cdse_covid.pegasus_pipeline.merge_aif.aif_models import Span


JUSTIFICATIONS = [
    URIRef('https://raw.githubusercontent.com/NextCenturyCorporation/AIDA-Interchange-Format/master/java/src/main/resources/com/ncc/aif/ontologies/InterchangeOntology#informativeJustification'),
    URIRef('https://raw.githubusercontent.com/NextCenturyCorporation/AIDA-Interchange-Format/master/java/src/main/resources/com/ncc/aif/ontologies/InterchangeOntology#justifiedBy'),
    URIRef('https://raw.githubusercontent.com/NextCenturyCorporation/AIDA-Interchange-Format/master/java/src/main/resources/com/ncc/aif/ontologies/InterchangeOntology#TextJustification')
]

SOURCE_DOC = URIRef('https://raw.githubusercontent.com/NextCenturyCorporation/AIDA-Interchange-Format/master/java/src/main/resources/com/ncc/aif/ontologies/InterchangeOntology#sourceDocument')

def get_claim_semantics_for_claim(graph: Graph, claim: URIRef) -> Optional[Sequence[URIRef]]:
    """Get Claim Semantics for a given Claim."""
    claim_semantics_uri_result = set(
        graph.query(
            """
        SELECT ?cs
        WHERE {
            ?claim aida:claimSemantics ?cs .
        }
        """,
            initBindings={"claim": claim[0]},
        )
    )

    if claim_semantics_uri_result:
        return list(claim_semantics_uri_result.pop())

    return None


def get_claims(graph: Graph) -> Set[URIRef]:
    """Get all claims on document graph."""
    return set(graph.query("SELECT ?c WHERE { ?c a aida:Claim . }"))


def get_span_for_claim(graph: Graph, claim: URIRef, *, provenance: str) -> Optional[Span]:
    """Get span for given Claim."""
    span_info = set(
        graph.query(
            """
            SELECT ?confidence ?end ?start ?source
            WHERE {
                ?c aida:justifiedBy ?justifiedBy .

                ?justifiedBy aida:endOffsetInclusive ?end ;
                    aida:startOffset ?start ;
                    aida:source ?source .

                OPTIONAL {
                    ?justifiedBy aida:confidence ?confb .

                    ?confb aida:confidenceValue ?confidence .
                }
            }
            """,
            initBindings={"c": claim[0]},
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
        logging.warning("No 'justifiedBy' element found for claim: %s", claim)
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
    ]:
        if combo in keys:
            return combo

    return None


def link_ent(graph: Graph, uiuc_claim: URIRef, ent: URIRef) -> Graph:
    """Link entity to graph."""
    update_cs = """
        INSERT { ?claim aida:claimSemantics ?semantics . }
        WHERE { ?claim aida:claimSemantics ?s . }
    """

    graph.update(update_cs, initBindings={"claim": uiuc_claim[0], "semantics": ent})

    update_kes = """
        INSERT { ?claim aida:associatedKEs ?semantics . }
        WHERE { ?claim aida:associatedKEs ?s . }
    """

    graph.update(update_kes, initBindings={"claim": uiuc_claim[0], "semantics": ent})

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



def get_all_event_types_and_arguments(graph: Graph) -> Set[URIRef]:
    query = """
    SELECT DISTINCT ?et
    WHERE { ?et a rdf:Statement }
    """
    return set(graph.query(query))


def is_in_updated_graph(graph: Graph, elem: URIRef) -> bool:
    query = """
    SELECT DISTINCT ?elem
    WHERE { 
        ?elem rdf:subject ?subject .
        ?subject a aida:Event .
    }
    """
    return set(graph.query(query, initBindings={"elem": elem}))



def get_source_id(graph: Graph, claim: URIRef) -> Set[URIRef]:
    query = """
    SELECT DISTINCT ?source
    WHERE {
        ?claim aida:justifiedBy ?jb .

        ?jb aida:sourceDocument ?source .
    }
    """
    res = set(graph.query(query, initBindings={"claim": claim}))
    if res:
        return res.pop()

def main(isi_store: Path, uiuc_store: Path, output: Path) -> None:
    """Entrypoint to Merge script."""
    isi_graphs = load_from_key_value_store(isi_store)
    uiuc_graphs = load_from_key_value_store(uiuc_store)

    for graph_id, uiuc_graph in tqdm(uiuc_graphs.items()):

        potential_isi_graph = isi_graphs.get(graph_id)
        if potential_isi_graph:

            isi_claims = get_claims(potential_isi_graph)
            isi_claims_with_spans = {}
            for claim in isi_claims:
                span = get_span_for_claim(potential_isi_graph, claim, provenance="ISI")
                if span:
                    isi_claims_with_spans[(span.start, span.end)] = claim

            uiuc_claims = get_claims(uiuc_graph)
            uiuc_claims_with_spans = {}
            for claim in uiuc_claims:
                span = get_span_for_claim(uiuc_graph, claim, provenance="UIUC")
                if span:
                    uiuc_claims_with_spans[(span.start, span.end)] = claim

            for uiuc_claim_span, uiuc_claim in uiuc_claims_with_spans.items():
                source_id = get_source_id(uiuc_graph, uiuc_claim[0])

                match = fuzzy_match(uiuc_claim_span, isi_claims_with_spans.keys())
                if match:
                    isi_claim = isi_claims_with_spans[match]
                    # get all claim semantics URIs (events & arguments)
                    claim_semantics = get_claim_semantics_for_claim(potential_isi_graph, isi_claim)

                    if claim_semantics:
                        for ent in claim_semantics:
                            # link ent
                            uiuc_graph = link_ent(uiuc_graph, uiuc_claim, ent)

                            # get all properties of said events/args
                            all_props = get_all_properties(potential_isi_graph, ent)
                            # create ent
                            uiuc_graph = create_ent(uiuc_graph, ent, all_props)

                            stack = list(all_props)
                            # dfs through all the properties to update graph
                            while stack:
                                curr_elem = stack.pop()
                                curr_elem_value = curr_elem[1]

                                prop_type = curr_elem[0]

                                all_props_of_curr = get_all_properties(
                                    potential_isi_graph, curr_elem_value
                                )

                                if prop_type in JUSTIFICATIONS:
                                    # breakpoint()
                                    all_props_of_curr.add((SOURCE_DOC, source_id[0]))

                                if all_props_of_curr:
                                    uiuc_graph = create_ent(
                                        uiuc_graph, curr_elem_value, all_props_of_curr
                                    )
                                    stack.extend(all_props_of_curr)

            all_event_types_and_args = get_all_event_types_and_arguments(potential_isi_graph)
            for elem in all_event_types_and_args:
                if is_in_updated_graph(uiuc_graph, elem):
                    all_props = get_all_properties(potential_isi_graph, elem)
                    if all_props:
                        uiuc_graph = create_ent(uiuc_graph, elem, all_props)

            uiuc_graph.serialize(output / graph_id, format="turtle")

    logging.info("Serialized merged graphs to %s", args.output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--isi-store", type=Path)
    p.add_argument("--uiuc-store", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    main(args.isi_store, args.uiuc_store, args.output)
