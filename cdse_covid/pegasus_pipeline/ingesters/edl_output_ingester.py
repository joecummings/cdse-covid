"""Ingest EDL documents and convert them to workable internal format."""
import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Dict, List, MutableMapping, Optional, Tuple


@dataclass
class EDLEntity:
    """Class for EDL Entity."""

    ent_id: str
    ent_type: str
    ent_link: Optional[str] = None
    type_link: Optional[str] = None


@dataclass
class EDLMention:
    """Class for EDL Mention."""

    doc_id: str
    text: str
    mention_type: str
    span: Tuple[int, int]
    parent_entity: EDLEntity


def main(edl_output: Path, output: Path) -> None:
    """Run EDL Ingester."""
    mention_store: MutableMapping[str, MutableMapping[Tuple[int, int], EDLMention]] = defaultdict(
        dict
    )
    set_aside_mentions: List[Tuple[str, str, str, str, Tuple[int, int]]] = []
    with open(edl_output, "r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        all_entities: Dict[str, EDLEntity] = {}
        # Collect all entities before looking for mentions
        for line in reader:
            if len(line) == 3:  # We have ourselves an entity
                _id = line[0].split("_")
                formatted_id = _id[-1]
                ent_link = None
                type_link = None
                ent_type = ""

                if line[1] == "link" and line[2].startswith("Q"):
                    ent_link = line[2]
                    new_entity = EDLEntity(
                        ent_id=formatted_id, ent_type=ent_type, ent_link=ent_link
                    )
                elif line[1] == "typelink" and line[2].startswith("Q"):
                    type_link = line[2]
                    new_entity = EDLEntity(
                        ent_id=formatted_id, ent_type=ent_type, type_link=type_link
                    )
                elif line[1] == "type":
                    # Type example: `https://tac.nist.gov/tracks/SM-KBP/2019/ontologies/LDCOntology#GPE.City`
                    # Sometimes a number gets included, separated by whitespace
                    ent_type_uri = line[2].split()[0]
                    ent_type = ent_type_uri.split("#")[-1]
                    new_entity = EDLEntity(ent_id=formatted_id, ent_type=ent_type)

                if all_entities.get(formatted_id):
                    # The entity already exists in our collection, so we just modify it
                    if ent_link:
                        all_entities[formatted_id].ent_link = ent_link
                    elif type_link:
                        all_entities[formatted_id].type_link = type_link
                    elif ent_type:
                        all_entities[formatted_id].ent_type = ent_type
                else:
                    all_entities[formatted_id] = new_entity

            elif len(line) == 5:  # Otherwise it's a mention
                _ent_id = line[0].split("_")
                formatted_ent_id = _ent_id[-1]
                mention_type = line[1]
                text = line[2]
                doc_and_span = line[3].split(":")
                doc = doc_and_span[0]
                _span = doc_and_span[1].split("-")
                formatted_span = (int(_span[0]), int(_span[1]))
                # Entities for mentions do not have their Qnode links yet.
                # Set aside the mention for examination
                # until after all entity data is collected.
                set_aside_mentions.append(
                    (formatted_ent_id, doc, text, mention_type, formatted_span)
                )

    # Now add the mentions set aside
    for (ent_id, doc, text, mention_type, span) in set_aside_mentions:
        new_mention = EDLMention(
            doc_id=doc,
            text=text,
            mention_type=mention_type,
            span=span,
            parent_entity=all_entities[ent_id],
        )
        mention_store[doc][span] = new_mention

    with open(output, "wb+") as handle:  # type: ignore
        pickle.dump(mention_store, handle)  # type: ignore


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--edl-output", help="Path to EDL output in ColdStart format", type=Path)
    p.add_argument("--output", help="Path to output file", type=Path)
    args = p.parse_args()
    main(args.edl_output, args.output)
