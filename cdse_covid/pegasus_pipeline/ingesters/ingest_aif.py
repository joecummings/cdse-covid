"""Ingest AIF documents."""
import argparse
import logging
from pathlib import Path

from pegasus_wrapper.key_value import ZipKeyValueStore
from rdflib import Graph, URIRef
from tqdm import tqdm
from vistautils.key_value import byte_key_value_sink_from_params
from vistautils.parameters import Parameters

from cdse_covid.pegasus_pipeline.merge_aif.merge_aif import get_claims

logger = logging.getLogger()


def get_doc_id(graph: Graph, claim: URIRef) -> str:
    """Get ID of document."""
    doc_id_result = set(
        graph.query(
            """
        SELECT ?doc_id
        WHERE {
            ?c aida:justifiedBy ?justifiedBy .

            ?justifiedBy aida:endOffsetInclusive ?end ;
                aida:startOffset ?start ;
                aida:source ?doc_id .
        }
        """,
            initBindings={"c": claim[0]},
        )
    )

    # If there's no justification, there must be a sourceDocument
    if not doc_id_result:
        doc_id_result = set(
            graph.query(
                """
            SELECT ?doc_id
            WHERE {
                ?c aida:sourceDocument ?doc_id
            }
            """,
                initBindings={"c": claim[0]},
            )
        )
        doc_id = doc_id_result.pop()
        return str(doc_id[0].value) + ".ttl"

    doc_id = doc_id_result.pop()
    return str(doc_id[0].value) + ".ttl"


def save_to_key_value_store(aif_dir: Path, kvs_path: Path, *, is_uiuc: bool) -> ZipKeyValueStore:
    """Save AIF to a ZipKeyValueStore."""
    local_params = Parameters.from_mapping({"output": {"type": "zip", "path": str(kvs_path)}})

    with byte_key_value_sink_from_params(local_params) as sink:
        for aif_file in tqdm(
            aif_dir.glob("*.ttl")
        ):  # TQDM doesn't actually work b/c it's a generator
            g = Graph()
            g = g.parse(source=aif_file, format="turtle")
            key = aif_file.name
            if is_uiuc:
                claims = get_claims(g)
                doc_ids = set()
                if not claims:
                    continue
                for claim in claims:
                    doc_id = get_doc_id(g, claim)
                    doc_ids.add(doc_id)
                if len(doc_ids) > 1:
                    raise RuntimeError(f"Claims from multiple documents found in {aif_file.name}")
                else:
                    key = doc_ids.pop()
            sink[key] = g.serialize(format="turtle", encoding="utf-8")

    return ZipKeyValueStore(path=kvs_path)


def main(aif_dir: Path, aif_as_zip: Path, *, is_uiuc: bool) -> None:
    """Entrypoint to AIF ingest script."""
    save_to_key_value_store(aif_dir, aif_as_zip, is_uiuc=is_uiuc)
    logger.info("Serialized all TTL files to: %s", aif_as_zip)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aif-dir", type=Path, help="option help")
    p.add_argument("--aif-as-zip", type=Path)
    p.add_argument("--is-uiuc", action="store_true", default=False)
    args = p.parse_args()
    main(args.aif_dir, args.aif_as_zip, is_uiuc=args.is_uiuc)
