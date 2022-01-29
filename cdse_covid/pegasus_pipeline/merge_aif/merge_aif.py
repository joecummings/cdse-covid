import argparse
import logging
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Set, Tuple

from rdflib import Graph, URIRef
from tqdm import tqdm
from vistautils.key_value import byte_key_value_source_from_params
from vistautils.parameters import Parameters

from cdse_covid.pegasus_pipeline.merge_aif.aif_models import Span


def get_claim_semantics_for_claim(graph: Graph, claim: URIRef) -> Optional[Sequence[URIRef]]:
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
        return claim_semantics_uri_result.pop()

    return None


def get_claims(graph: Graph) -> Set[URIRef]:
    return set(graph.query("SELECT ?c WHERE { ?c a aida:Claim . }"))


def get_span_for_claim(graph: Graph, claim: URIRef, *, provenance: str) -> Optional[Span]:
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
            start=span_info_dict["start"],
            end=span_info_dict["end"],
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


def fuzzy_match(span: Tuple[int, int], keys: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
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

    update_query = """
        INSERT { ?claim aida:claimSemantics ?semantics . }
        WHERE { ?claim aida:claimSemantics ?s . }
    """

    graph.update(update_query, initBindings={"claim": uiuc_claim[0], "semantics": ent})

    return graph


def create_ent(graph: Graph, ent: URIRef, props: Set[Tuple[URIRef, URIRef]]) -> Graph:
    query = """
    INSERT {
        ?ent ?p ?v
    }
    WHERE {
        OPTIONAL { ?ent ?p ?v }
    }
    """

    for p, v in props:
        graph.update(query, initBindings={"ent": ent, "p": p, "v": v})

    return graph


def get_all_properties(graph: Graph, elem: URIRef) -> Set[Tuple[URIRef, URIRef]]:
    query = """
    SELECT DISTINCT ?property ?value
    WHERE {
        ?elem ?property ?value .
        OPTIONAL { ?elem a ?property . }
    }
    """
    return set(graph.query(query, initBindings={"elem": elem}))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--isi-store", type=Path)
    p.add_argument("--uiuc-store", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()

    isi_graphs = load_from_key_value_store(args.isi_store)
    uiuc_graphs = load_from_key_value_store(args.uiuc_store)

    for graph_id, uiuc_graph in tqdm(uiuc_graphs.items()):

        potential_isi_graph = isi_graphs.get(graph_id)
        if potential_isi_graph:

            isi_claims = get_claims(potential_isi_graph)
            isi_claims_with_spans = {}
            for claim in isi_claims:
                x = get_span_for_claim(potential_isi_graph, claim, provenance="ISI")
                isi_claims_with_spans[(x.start.value, x.end.value)] = claim

            uiuc_claims = get_claims(uiuc_graph)
            uiuc_claims_with_spans = {}
            for claim in uiuc_claims:
                x = get_span_for_claim(uiuc_graph, claim, provenance="UIUC")
                uiuc_claims_with_spans[(x.start.value, x.end.value)] = claim

            for uiuc_claim_span, uiuc_claim in uiuc_claims_with_spans.items():
                match = fuzzy_match(uiuc_claim_span, list(isi_claims_with_spans.keys()))
                if match:
                    isi_claim = isi_claims_with_spans[match]
                    # get all claim semantics URIs (events & arguments)
                    claim_semantics = get_claim_semantics_for_claim(potential_isi_graph, isi_claim)

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
                            all_props_of_curr = get_all_properties(
                                potential_isi_graph, curr_elem_value
                            )
                            if all_props_of_curr:
                                uiuc_graph = create_ent(
                                    uiuc_graph, curr_elem_value, all_props_of_curr
                                )
                                stack.extend(all_props_of_curr)

            uiuc_graph.serialize(args.output / f"{graph_id}-merged.ttl", format="turtle")

    logging.info("Serialized merged graphs to %s", args.output)


if __name__ == "__main__":
    main()
