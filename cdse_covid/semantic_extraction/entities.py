from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple


@dataclass
class Entity:
    _id: str


@dataclass
class AMRLabel(Entity):
    graph: Any
    alignments: Any


@dataclass
class XVariable(Entity):
    span: Tuple[int, int]
    text: str


@dataclass
class Claimer(Entity):
    span: Tuple[int, int]
    text: str


@dataclass
class SRLabel:
    verb: str
    args: Mapping[str, str]


@dataclass
class WikidataQnode:
    label: str
    description: Optional[str] = None
    from_query: Optional[str] = None
