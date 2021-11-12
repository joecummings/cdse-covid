from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple


@dataclass
class Entity:
    ent_id: str
    span: Tuple[int, int]
    text: str


@dataclass
class AMREntity(Entity):
    graph: Any
    alignments: Any


@dataclass
class XVariable(Entity):
    pass


@dataclass
class Claimer(Entity):
    pass


@dataclass
class SRLabel:
    verb: str
    args: Mapping[str, str]


@dataclass
class WikidataQnode:
    label: str
    description: Optional[str] = None
    from_query: Optional[str] = None
