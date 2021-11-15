"""Collection of Entities."""
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple


@dataclass
class Entity:
    """Base entity."""

    text: Optional[str] = ""
    ent_id: Optional[int] = None
    doc_id: Optional[str] = None
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

    qnode_id: Optional[str] = None
    description: Optional[str] = None
    from_query: Optional[str] = None


@dataclass
class ClaimEvent(WikidataQnode):
    """Claim event entity."""


@dataclass
class ClaimArg(WikidataQnode):
    """Claim arg entity."""


@dataclass
class ClaimSemantics:
    """Claim Semantics entity."""

    event: Optional[ClaimEvent] = None
    args: Optional[Mapping[str, ClaimArg]] = None
