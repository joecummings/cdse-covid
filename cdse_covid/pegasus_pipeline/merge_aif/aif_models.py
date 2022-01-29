from rdflib.term import Node


class MergeNode(Node):
    def __init__(self, provenance: str = "") -> None:
        super().__init__()
        self.provenance = provenance

    def __repr__(self) -> str:
        return str(self.__dict__)


class Span(MergeNode):
    def __init__(self, start: int, end: int, source: str = "", provenance: str = "") -> None:
        super().__init__(provenance=provenance)
        self.start = start
        self.end = end
        self.source = source
