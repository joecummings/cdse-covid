import abc
import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

from allennlp_models.pretrained import load_predictor
from common import SENT_MODEL
from dataset import AIDADataset
import numpy as np
from sentence_transformers import SentenceTransformer, util
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span
from srl_utils import clean_srl_output
import torch
from wikidata_linker.wikidata_linking import disambiguate_kgtk

logging.getLogger("ROOT").setLevel(logging.WARNING)

CORONA_NAME_VARIATIONS = [
    "COVID-19",
    "Coronavirus",
    "SARS-CoV-2",
    "coronavirus",
    "corona virus",
    "covid-19",
    "covid 19",
]

NLP = spacy.load("en_core_web_sm")
SRLModel = load_predictor("structured-prediction-srl")

RegexPattern = Sequence[Mapping[str, Any]]

TEMPLATE_FILE = Path(__file__).parent / "topic_list.txt"


class ClaimDetector:
    def _structure_gaia_claim(
        self, *, span: Span, rule_id: str, x_variable: str, claimer: str
    ) -> Mapping[str, str]:
        return {
            "Natural Language Description": span.sent.text,
            "Claim Template": rule_id,
            "X Variable": x_variable,
            "Claimer": claimer,
            "Epistemic Status": "true-certain",
        }

    @abc.abstractmethod
    def find_matches(self, corpus: List[Path]):
        pass

    @abc.abstractmethod
    def save(self, out_file: Path) -> Path:
        pass


class SBERTClaimDetector(ClaimDetector):
    def __init__(self) -> None:
        self.ss_model = SentenceTransformer(str(SENT_MODEL))

    def find_matches(self, corpus: List[Path]):
        all_docs = AIDADataset(nlp=NLP, templates_file=TEMPLATE_FILE)

        encoded_docs = []
        for doc in all_docs._batch_convert_to_spacy(corpus):
            doc_sentences = []
            for sent in doc.sents:
                encoded_sent = self.ss_model.encode(
                    sent.text, convert_to_tensor=True, normalize_embeddings=True
                )
                doc_sentences.append(encoded_sent)
            encoded_docs.append(doc_sentences)

        encoded_templates = []
        for template in all_docs.templates:
            encoded_template = self.ss_model.encode(
                template, convert_to_tensor=True, normalize_embeddings=True
            )
            encoded_templates.append(encoded_template)

        all_template_doc_sims = []
        for template in encoded_templates:
            template_sim_scores = []
            for doc in encoded_docs:
                doc_sim_scores = torch.tensor([util.dot_score(sent, template) for sent in doc])
                template_sim_scores.append(doc_sim_scores)
            all_template_doc_sims.append(template_sim_scores)

        for pair in all_template_doc_sims:
            for sentence in pair:
                # top_3 = sentence.argmax(3)
                import pdb
                pdb.set_trace()

        
                


    def save(self, out_file: Path) -> Path:
        pass


class RegexClaimDetector(ClaimDetector, Matcher):
    def __init__(self) -> None:
        Matcher.__init__(self, NLP.vocab, validate=True)

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

    def _find_links(self, sentence: Span, tokens: Sequence[str]):
        """Find WikiData links for a set of tokens."""
        return (disambiguate_kgtk(sentence.text, token, thresh=0.0) for token in tokens)

    def _label_roles(self, span_text: str) -> Mapping[str, str]:
        """Get semantic role labels for text.

        Args:
            span_text: Text of the current span

        Return:
            Mapping of role -> text
        """
        roles = SRLModel.predict(span_text)
        if len(roles["verbs"]) > 1:
            logging.warning("More than one main verb in instance: %s", span_text)
        return clean_srl_output(roles, NLP)

    def _identify_claimer(self, span):
        """Identify the claimer of the span."""
        raise NotImplementedError()

    def find_matches(self, corpus: Path) -> Mapping[str, str]:
        all_docs = AIDADataset(
            nlp=NLP, templates_file=TEMPLATE_FILE
        )._batch_convert_to_spacy(corpus)

        all_matches = []
        for doc in all_docs:
            matches = self.__call__(doc)
            for (match_id, start, end) in matches:
                rule_id: str = doc.vocab.strings[match_id]
                span = doc[start:end]
                # Expand to get entire noun phrases

                # Label w/ SRL
                match_roles = self._label_roles(span.text)

                cleaned_tokens = []
                for token in rule_id.split():
                    if "-" in token and "X" in token:
                        cleaned_tokens.append("X")
                    else:
                        cleaned_tokens.append(token)
                cleaned_rule_id = " ".join(cleaned_tokens)

                template_roles = self._label_roles(cleaned_rule_id)
                target_role = None
                for role, token in template_roles.items():
                    if token == "X":
                        target_role = role

                # Identify Claimer
                claimer = ""

                # Link & Format QNodes
                links = self._find_links(span.sent, match_roles.values())
                x_variable = match_roles.get(target_role, "")
                for link in links:
                    if link["query"] == x_variable and len(link["options"]) > 0:
                        x_variable += f" ({link['options'][0]['qnode']})"

                # Add to returns with proper claim format
                claim = self._structure_gaia_claim(
                    span=span,
                    rule_id=rule_id,
                    x_variable=x_variable,
                    claimer=claimer,
                )
                all_matches.append(claim)

        return all_matches

    @staticmethod
    def save(out_file: Path, matches: Sequence[Mapping[str, Any]]) -> None:
        with open(out_file, "w+") as handle:
            writer = csv.writer(handle)
            count = 0
            for match in matches:
                if count == 0:
                    header = match.keys()
                    writer.writerow(header)
                    count += 1
                writer.writerow(match.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Input docs", type=Path)
    parser.add_argument(
        "--patterns", help="Patterns for Regex", type=Path, default=None
    )
    parser.add_argument("--out", help="Out file", type=Path)
    args = parser.parse_args()

    if args.patterns:
        with open(args.patterns, "r") as handle:
            patterns = json.load(handle)

    matcher = RegexClaimDetector()
    matcher.add_patterns(patterns)
    matches = matcher.find_matches(args.input)
    matcher.save(args.out, matches)

    # matcher = SBERTClaimDetector()
    # matches = matcher.find_matches(args.input)



if __name__ == "__main__":
    main()
