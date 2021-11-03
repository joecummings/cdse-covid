from collections import defaultdict
import csv
from dataclasses import dataclass, field, replace
from pathlib import Path
import pickle
from typing import Any, Mapping, Optional, Sequence, Tuple
from cdse_covid.semantic_extraction.models import WikidataQnode
import logging


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
    claim_semantics: Optional[Mapping[str, Any]] = None
    epistemic_status: Optional[bool] = None
    sentiment_status: Optional[str] = None
    theories: Mapping[str, Any] = field(default_factory=dict)

    def add_theory(self, name: str, theory: Any) -> None:
        self.theories[name] = theory

    def get_theory(self, name: str) -> Optional[Any]:
        return self.theories.get(name)
    
    @staticmethod
    def to_json(obj, classkey=None):
        if isinstance(obj, dict):
            data = {k: Claim.to_json(v, classkey) for (k, v) in obj.items()}
            return data
        elif hasattr(obj, "_ast"):
            return Claim.to_json(obj._ast())
        elif hasattr(obj, "__iter__") and not isinstance(obj, str):
            return [Claim.to_json(v, classkey) for v in obj]
        elif hasattr(obj, "__dict__"):
            data = dict([(key, Claim.to_json(value, classkey)) 
                for key, value in obj.__dict__.items() 
                if not callable(value) and not key.startswith('_')])
            if classkey is not None and hasattr(obj, "__class__"):
                data[classkey] = obj.__class__.__name__
            return data
        else:
            return obj


class ClaimDataset:
    def __init__(self, claims: Sequence[Claim] = []) -> None:
        self.claims = claims

    def add_claim(self, claim: Claim):
        if not self.claims:
            self.claims = []
        self.claims.append(claim)

    def __iter__(self):
        return iter(self.claims)

    @staticmethod
    def from_multiple_claims_ds(*claim_datasets: "ClaimDataset") -> "ClaimDataset":
        datasets_dict = defaultdict(list)
        for dataset in claim_datasets:
            for claim in dataset:
                datasets_dict[claim.claim_id].append(claim)

        all_claims = []
        for _, claims in datasets_dict.items():
            for i, claim in enumerate(claims):
                if i == 0:
                    new_claim = claim
                else:
                    all_non_none_attrs = {k:v for k,v in claim.__dict__.items() if v != None and k != "theories"}
                    new_claim: Claim = replace(new_claim, **all_non_none_attrs)
                    for k, v in claim.theories.items():
                        if not new_claim.get_theory(k):
                            new_claim.add_theory(k, v)
            all_claims.append(new_claim)
        return ClaimDataset(all_claims)


    @staticmethod
    def load_from_dir(path: Path) -> "ClaimDataset":
        claims = []
        for claim_file in path.glob("*.claim"):
            with open(claim_file, "rb") as handle:
                claim = pickle.load(handle)
                claims.append(claim)
        return ClaimDataset(claims)

    def save_to_dir(self, path: Path):
        if not self.claims:
            logging.warning("No claims found.")
        path.mkdir(exist_ok=True)
        for claim in self.claims:
            with open(f"{path / str(claim.claim_id)}.claim", "wb+") as handle:
                pickle.dump(claim, handle, pickle.HIGHEST_PROTOCOL)

    def print_out_claim_sentences(self, path: Path):
        if not self.claims:
            logging.warning("No claims found.")
        with open(path, "w+") as handle:
            writer = csv.writer(handle)
            for claim in self.claims:
                writer.writerow([claim.claim_text])