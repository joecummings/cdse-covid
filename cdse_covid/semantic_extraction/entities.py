"""Collection of Entities."""
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Entity:
    """Base entity."""

    text: Optional[str] = ""
    ent_id: Optional[int] = None
    span: Optional[Tuple[int, int]] = None


@dataclass
class XVariable(Entity):
    """XVariable entity."""


@dataclass
class Claimer(Entity):
    """Claimer entity."""


@dataclass
class WikidataQnode(Entity):
    """Qnode entity."""

    description: Optional[str] = None
    from_query: Optional[str] = None
