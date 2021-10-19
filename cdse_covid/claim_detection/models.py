from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple
from cdse_covid.semantic_extraction.models import WikidataQnode


@dataclass
class Claim:
    claim_id: int
    doc_id: str
    claim_text: str
    claim_sentence: str
    claim_span: Tuple[str, str]
    claim_template: Optional[str] = None
    topic: Optional[str] = None
    subtopic: Optional[str] = None
    x_variable: Optional[str] = None
    x_variable_qnode: Optional[WikidataQnode] = None
    claimer: Optional[str] = None
    claimer_qnode: Optional[WikidataQnode] = None
    claim_date_time: Optional[str] = None
    claim_location: Optional[str] = None
    claim_location_qnode: Optional[WikidataQnode] = None
    event_qnode: Optional[WikidataQnode] = None
    epistemic_status: Optional[bool] = None
    sentiment_status: Optional[str] = None
    theories: Mapping[str, Any] = field(default_factory=dict)

    def add_theory(self, name: str, theory: Any) -> None:
        self.theories[name] = theory

    def get_theory(self, name: str) -> Any:
        return self.theories[name]