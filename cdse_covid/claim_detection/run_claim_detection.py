"""Run claim detection over corpus of SpaCy encoded documents."""
import abc
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence
import pickle
import spacy
from spacy.language import Language
from spacy.matcher import Matcher
from cdse_covid.dataset import AIDADataset
from spacy.tokens import Span
import uuid
from collections import defaultdict
import csv
from cdse_covid.claim_detection.models import Claim
from dataclasses import replace

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


class ClaimDataset:
    def __init__(self, claims: Sequence[Claim] = []) -> None:
        self.claims = claims

    def add_claim(self, claim: Claim):
        if not self.claims:
            self.claims = []
        self.claims.append(claim)

    def __iter__(self) -> Iterable[Claim]:
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


    @classmethod
    def load_from_dir(cls, path: Path) -> "ClaimDataset":
        claims = []
        for claim_file in path.glob("*.claim"):
            with open(claim_file, "rb") as handle:
                claim = pickle.load(handle)
                claims.append(claim)
        return cls(claims)

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
                    claim_text=span.text,
                    claim_sentence=span.sent.text,
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

    topics_info = {}
    with open(Path(__file__).parent / "topic_list.txt", "r") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            key = row[3]
            topics_info[key] = {"topic": row[2], "subtopic": row[1]}
    
    for claim in matches:
        template = topics_info[claim.claim_template]
        claim.topic = template["topic"]
        claim.subtopic = template["subtopic"]
                
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

    main(args.input, args.patterns, args.out, spacy_model=model)
