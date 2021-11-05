import abc
from typing import Any, Mapping, Optional, Sequence
from uuid import uuid1
from spacy.matcher import Matcher
from cdse_covid.claim_detection.models import Claim, ClaimDataset
from cdse_covid.dataset import AIDADataset
from spacy.tokens import Span

RegexPattern = Sequence[Mapping[str, Any]]


CORONA_NAME_VARIATIONS = [
    "COVID-19",
    "Coronavirus",
    "SARS-CoV-2",
    "coronavirus",
    "corona virus",
    "covid-19",
    "covid 19",
]

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
                    claim_id=int(uuid1()),
                    doc_id=doc[0],
                    claim_text=span.text,
                    claim_sentence=span.sent.text,
                    claim_span=(start, end),
                    claim_template=rule_id,
                )

                claim_dataset.add_claim(new_claim)

        return claim_dataset