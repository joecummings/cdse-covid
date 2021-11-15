"""Claim module."""
from dataclasses import dataclass, field
from typing import Any, List, MutableMapping, Optional, Tuple, Union

from cdse_covid.semantic_extraction.entities import (
    Claimer,
    ClaimSemantics,
    WikidataQnode,
    XVariable,
)


@dataclass
class Claim:
    """A claim as documented by GAIA Nov 12, 2021."""

    claim_id: int
    doc_id: str
    claim_text: str
    claim_sentence: str
    claim_span: Tuple[str, str]
    claim_template: Optional[str] = None
    topic: Optional[str] = None
    subtopic: Optional[str] = None
    x_variable: Optional[XVariable] = None
    x_variable_qnode: Optional[WikidataQnode] = None
    claimer: Optional[Claimer] = None
    claimer_qnode: Optional[WikidataQnode] = None
    claim_date_time: Optional[str] = None
    claim_location: Optional[str] = None
    claim_location_qnode: Optional[WikidataQnode] = None
    claim_semantics: Optional[ClaimSemantics] = None
    epistemic_status: Optional[bool] = None
    sentiment_status: Optional[str] = None
    theories: MutableMapping[str, Any] = field(default_factory=dict)

    def add_theory(self, name: str, theory: Any) -> None:
        """Add a theory to the claim obj."""
        self.theories[name] = theory

    def get_theory(self, name: str) -> Optional[Any]:
        """Get an existing theory by *name*."""
        return self.theories.get(name)

    @staticmethod
    def to_json(
        obj: Any, classkey: Optional[str] = None
    ) -> Union[List[MutableMapping[str, Any]], MutableMapping[str, Any], Any]:
        """Convert claim obj into a JSON mapping."""
        if isinstance(obj, dict):
            data = {k: Claim.to_json(v, classkey) for (k, v) in obj.items()}
            return data
        elif hasattr(obj, "_ast"):
            return Claim.to_json(obj._ast())  # pylint: disable=protected-access
        elif hasattr(obj, "__iter__") and not isinstance(obj, str):
            return [Claim.to_json(v, classkey) for v in obj]
        elif hasattr(obj, "__dict__"):
            data = dict(
                [
                    (key, Claim.to_json(value, classkey))
                    for key, value in obj.__dict__.items()
                    if not callable(value) and not key.startswith("_")
                ]
            )
            if classkey is not None and hasattr(obj, "__class__"):
                data[classkey] = obj.__class__.__name__
            return data
        else:
            return obj
