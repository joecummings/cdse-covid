from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Entity:
    text: Optional[str] = ""
    ent_id: Optional[int] = None
    span: Optional[Tuple[int, int]] = None


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
