"""Run claim detection over corpus of SpaCy encoded documents."""
import abc
import argparse
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple
import pickle
import spacy
from spacy.language import Language
from spacy.matcher import Matcher
from cdse_covid.dataset import AIDADataset
from spacy.tokens import Span
import uuid
from collections import defaultdict
import csv
from cdse_covid.semantic_extraction.run_wikidata_linking import WikidataQnode


CORONA_NAME_VARIATIONS = [
    "COVID-19",
    "Coronavirus",
    "SARS-CoV-2",
    "coronavirus",
    "corona virus",
    "covid-19",
    "covid 19",
]


RegexPattern = Sequence[Mapping[str, Any]]


@dataclass
class AMRLabel:
    label_id: str
    graph: Any
    alignments: Any



@dataclass
class Claim:
    claim_id: int
    doc_id: str
    text: str
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


class ClaimDataset:
    def __init__(self, claims: Sequence[Claim] = None) -> None:
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
        for claim_id, claims in datasets_dict.items():
            for i, claim in enumerate(claims):
                if i == 0:
                    new_claim = Claim(
                        claim_id,
                        claim.doc_id,
                        claim.text,
                        claim.claim_span,
                        claim.claim_template,
                        claim.theories
                    )
                else:
                    for name, theory in claim.theories.items():
                        new_claim.add_theory(name, theory)
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
                writer.writerow([claim.text])



class ClaimDetector:
    @abc.abstractmethod
    def generate_candidates(self, corpus: AIDADataset):
        pass


class RegexClaimDetector(ClaimDetector, Matcher):
    """Detect claims using Regex patterns."""

    def __init__(self, spacy_model) -> None:
        Matcher.__init__(self, spacy_model.vocab, validate=True)

    def add_patterns(self, patterns: Optional[Sequence[RegexPattern]] = None):
        """Add Regex patterns to the SpaCy Matcher."""
        for pattern, regexes in patterns.items():
            for regex in regexes:
                for row in regex:
                    possible_corona_name = row.get("TEXT")
                    if isinstance(possible_corona_name, dict):
                        possible_corona_name = possible_corona_name.get("IN")
                    if possible_corona_name == "CORONA_NAME_VARIATIONS":
                        row["TEXT"]["IN"] = CORONA_NAME_VARIATIONS
            self.add(pattern, regexes)

    def generate_candidates(self, corpus: AIDADataset, vocab) -> ClaimDataset:
        """Generate claim candidates from corpus based on Regex matches."""
        claim_dataset = ClaimDataset()

        for doc in corpus.documents:
            matches = self.__call__(doc[1])
            for (match_id, start, end) in matches:
                rule_id: str = vocab.strings[match_id]
                span: Span = doc[1][start:end]

                new_claim = Claim(
                    claim_id=int(uuid.uuid1()),
                    doc_id=doc[0],
                    text=span.sent.text,
                    claim_span=(start, end),
                    claim_template=rule_id,
                )

                claim_dataset.add_claim(new_claim)

        return claim_dataset


def main(input_corpus: Path, patterns: Path, out_dir: Path, *, spacy_model: Language):

    regex_model = RegexClaimDetector(spacy_model)
    with open(patterns, "r") as handle:
        patterns_json = json.load(handle)
    regex_model.add_patterns(patterns_json)

    dataset = AIDADataset.from_serialized_docs(input_corpus)
    matches = regex_model.generate_candidates(dataset, spacy_model.vocab)
    matches.save_to_dir(out_dir)

    logging.info("Saved matches to %s", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Input docs", type=Path)
    parser.add_argument(
        "--patterns", help="Patterns for Regex", type=Path, default=None
    )
    parser.add_argument("--out", help="Out file", type=Path)
    parser.add_argument("--spacy-model", type=Path)
    args = parser.parse_args()

    model = spacy.load(args.spacy_model)
    from cdse_covid.claim_detection.run_claim_detection import Claim # pylint: disable=import-self

    main(args.input, args.patterns, args.out, spacy_model=model)
