from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Entity:
    ent_id: str
    span: Tuple[int, int]
    text: str


@dataclass
class XVariable(Entity):
    pass


@dataclass
class Claimer(Entity):
    pass


@dataclass
class WikidataQnode(Entity):
    description: Optional[str] = None
    from_query: Optional[str] = None
