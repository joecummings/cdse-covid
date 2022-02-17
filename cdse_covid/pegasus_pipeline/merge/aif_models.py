"""AIF Models."""
from typing import Optional

from rdflib.term import Node


class MergeNode(Node):
    """Node to handle merge provenance."""

    def __init__(self, provenance: Optional[str] = "") -> None:
        """Init MergeNode."""
        super().__init__()
        self.provenance = provenance

    def __repr__(self) -> str:
        """Represent MergNode."""
        return str(self.__dict__)


class Span(MergeNode):
    """Node to handle Span."""

    def __init__(
        self, start: int, end: int, source: str = "", provenance: Optional[str] = ""
    ) -> None:
        """Init Span."""
        super().__init__(provenance=provenance)
        self.start = start
        self.end = end
        self.source = source
