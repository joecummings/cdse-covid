"""Run claim detection over corpus of SpaCy encoded documents."""
import abc
import argparse
from collections import defaultdict
import csv
from dataclasses import replace
import json
import logging
from pathlib import Path
import pickle
from typing import Any, Dict, List
import uuid

import spacy
from spacy.language import Language
from spacy.matcher import Matcher
from spacy.tokens import Span

from cdse_covid.claim_detection.claim import Claim
from cdse_covid.dataset import AIDADataset

CORONA_NAME_VARIATIONS = [
    "COVID-19",
    "Coronavirus",
    "SARS-CoV-2",
    "coronavirus",
    "corona virus",
    "covid-19",
    "covid 19",
]


RegexPattern = Dict[str, Any]


class ClaimDataset:
    """Dataset of claims."""

    def __init__(self, claims: List[Claim] = []) -> None:
        """Init ClaimDataset."""
        self.claims = claims

    def add_claim(self, claim: Claim) -> None:
        """Add a claim to the dataset."""
        if not self.claims:
            self.claims = []
        self.claims.append(claim)

    def __iter__(self) -> Any:
        """Return iterator over claims in dataset."""
        return iter(self.claims)

    @staticmethod
    def from_multiple_claims_ds(*claim_datasets: "ClaimDataset") -> "ClaimDataset":
        """Create a new instance of the dataset from multiple *claim_datasets*."""
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
                    all_non_none_attrs = {
                        k: v for k, v in claim.__dict__.items() if v is not None and k != "theories"
                    }
                    new_claim = replace(new_claim, **all_non_none_attrs)
                    for k, v in claim.theories.items():
                        if not new_claim.get_theory(k):
                            new_claim.add_theory(k, v)
            all_claims.append(new_claim)
        return ClaimDataset(all_claims)

    @staticmethod
    def load_from_dir(path: Path) -> "ClaimDataset":
        """Load a ClaimDataset from a directory of claims."""
        claims = []
        for claim_file in path.glob("*.claim"):
            with open(claim_file, "rb") as handle:
                claim = pickle.load(handle)
                claims.append(claim)
        return ClaimDataset(claims)

    def save_to_dir(self, path: Path) -> None:
        """Save all claims in dataset to given *path*."""
        if not self.claims:
            logging.warning("No claims found.")
        path.mkdir(exist_ok=True)
        for claim in self.claims:
            with open(f"{path / str(claim.claim_id)}.claim", "wb+") as handle:
                pickle.dump(claim, handle, pickle.HIGHEST_PROTOCOL)

    def print_out_claim_sentences(self, path: Path) -> None:
        """Print out the actual text of the existing claims."""
        if not self.claims:
            logging.warning("No claims found.")
        with open(path, "w+", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            for claim in self.claims:
                writer.writerow([claim.claim_text])


class ClaimDetector:
    """ABC to detect claims."""

    @abc.abstractmethod
    def generate_candidates(self, corpus: AIDADataset, *args: Any) -> ClaimDataset:
        """ABM for generating candidates."""


class RegexClaimDetector(ClaimDetector, Matcher):  # type: ignore
    """Detect claims using Regex patterns."""

    def __init__(self, spacy_model: Language) -> None:
        """Init RegexClaimDetector."""
        Matcher.__init__(self, spacy_model.vocab, validate=True)

    def add_patterns(self, patterns: RegexPattern) -> None:
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

    def generate_candidates(self, corpus: AIDADataset, *args: Any) -> ClaimDataset:
        """Generate claim candidates from corpus based on Regex matches."""
        claim_dataset = ClaimDataset()
        vocab = args[0]

        for doc in corpus.documents:
            matches = self.__call__(doc[1])
            for match_id, start, end in matches:
                rule_id: str = vocab.strings[match_id]
                span: Span = doc[1][start:end]
                span_offset_start = span.start_char
                span_offset_end = span.end_char

                # Get offsets from tokens in the claim's sentence
                claim_sentence_tokens_to_offsets = {}
                idx = span.sent.start_char
                for sentence_token in span.sent:
                    claim_sentence_tokens_to_offsets[sentence_token.text] = (
                        idx,
                        idx + len(sentence_token.text),
                    )
                    idx += len(sentence_token.text_with_ws)

                new_claim = Claim(
                    claim_id=int(uuid.uuid1()),
                    doc_id=doc[0],
                    claim_text=span.text,
                    claim_sentence=span.sent.text,
                    claim_span=(span_offset_start, span_offset_end),
                    claim_sentence_tokens_to_offsets=claim_sentence_tokens_to_offsets,
                    claim_template=rule_id,
                )

                claim_dataset.add_claim(new_claim)

        return claim_dataset


def main(input_corpus: Path, patterns: Path, out_dir: Path, *, spacy_model: Language) -> None:
    """Entrypoint to claim detection script."""
    regex_model = RegexClaimDetector(spacy_model)
    with open(patterns, "r", encoding="utf-8") as handle:
        patterns_json = json.load(handle)
    regex_model.add_patterns(patterns_json)

    dataset = AIDADataset.from_serialized_docs(input_corpus)
    matches = regex_model.generate_candidates(dataset, spacy_model.vocab)

    topics_info = {}
    with open(Path(__file__).parent / "topic_list.txt", "r", encoding="utf-8") as handle:
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
    parser.add_argument("--patterns", help="Patterns for Regex", type=Path, default=None)
    parser.add_argument("--out", help="Out file", type=Path)
    parser.add_argument("--spacy-model", type=Path)
    pargs = parser.parse_args()

    model = spacy.load(pargs.spacy_model)

    main(pargs.input, pargs.patterns, pargs.out, spacy_model=model)
