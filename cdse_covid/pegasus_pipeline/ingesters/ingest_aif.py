"""Ingest AIF documents."""
import argparse
import logging
from pathlib import Path
from typing import Optional

from pegasus_wrapper.key_value import ZipKeyValueStore
from rdflib import Graph
from tqdm import tqdm
from vistautils.key_value import byte_key_value_sink_from_params
from vistautils.parameters import Parameters

logger = logging.getLogger()


def get_abstract_doc_id(graph: Graph) -> Optional[str]:
    """Get doc id from any text justification element."""
    query = """
    SELECT ?doc_id
    WHERE {
        ?e a aida:TextJustification .

        ?e aida:source ?doc_id .
    }
    """
    doc_id = set(graph.query(query))
    if doc_id:
        doc_id_pair = tuple(doc_id.pop())
        return str(doc_id_pair[0].value) + ".ttl"

    return None


def save_to_key_value_store(aif_dir: Path, kvs_path: Path) -> ZipKeyValueStore:
    """Save AIF to a ZipKeyValueStore."""
    local_params = Parameters.from_mapping({"output": {"type": "zip", "path": str(kvs_path)}})

    with byte_key_value_sink_from_params(local_params) as sink:
        for aif_file in tqdm(
            aif_dir.glob("*.ttl")
        ):  # TQDM doesn't actually work b/c it's a generator
            g = Graph()
            g = g.parse(source=aif_file, format="turtle")

            key = get_abstract_doc_id(g)

            if key:
                sink[key] = g.serialize(format="turtle", encoding="utf-8")

    return ZipKeyValueStore(path=kvs_path)


def main(aif_dir: Path, aif_as_zip: Path) -> None:
    """Entrypoint to AIF ingest script."""
    save_to_key_value_store(aif_dir, aif_as_zip)
    logger.info("Serialized all TTL files to: %s", aif_as_zip)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aif-dir", type=Path, help="option help")
    p.add_argument("--aif-as-zip", type=Path)
    args = p.parse_args()
    main(args.aif_dir, args.aif_as_zip)
