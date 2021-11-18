"""Collection of Mentions."""
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple


@dataclass
class Mention:
    """Base mention."""

    text: Optional[str] = ""
    mention_id: Optional[int] = None
    ent_id: Optional[str] = None
    doc_id: Optional[str] = None
    span: Optional[Tuple[int, int]] = None


@dataclass
class XVariable(Mention):
    """XVariable mention."""


@dataclass
class Claimer(Mention):
    """Claimer mention."""


@dataclass
class WikidataQnode(Mention):
    """Qnode mention."""

    qnode_id: Optional[str] = None
    description: Optional[str] = None
    from_query: Optional[str] = None


@dataclass
class ClaimEvent(WikidataQnode):
    """Claim event mention."""


@dataclass
class ClaimArg(WikidataQnode):
    """Claim arg mention."""


@dataclass
class ClaimSemantics:
    """Claim Semantics mention."""

    event: Optional[ClaimEvent] = None
    args: Optional[Mapping[str, ClaimArg]] = None
