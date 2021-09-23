import abc
import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

from allennlp_models.pretrained import load_predictor
from sentence_transformers import SentenceTransformer, util
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span

from wikidata_linker.wikidata_linking import disambiguate_kgtk

logging.getLogger("allennlp.nn.initializers").setLevel(logging.WARNING)
logging.getLogger("allennlp.common.params").setLevel(logging.WARNING)
logging.getLogger("allennlp.*").setLevel(logging.WARNING)

CORONA_NAME_VARIATIONS = [
    "COVID-19",
    "Coronavirus",
    "SARS-CoV-2",
    "coronavirus",
    "corona virus",
    "covid-19",
    "covid 19",
]
# ss_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

nlp = spacy.load("en_core_web_sm")
srl = load_predictor("structured-prediction-srl")

REGEX_PATTERN = Sequence[Mapping[str, Any]]


def clean_srl_output(srl_output: Mapping[str, Any]) -> Mapping[str, str]:
    """Takes AllenNLP SRL output and returns list of ARGN strings.

    Args:
        srl_output: A string of semantic role labelling output

    Returns:
        A list of ARG strings from SRL output.
    """
    cleaned_output = []
    for verb in srl_output["verbs"]:
        if "tags" in verb and len(verb["tags"]) > 0:
            tag_sequences: Mapping[str, str] = {}
            tags_words = zip(
                [tag.split("-", 1)[-1] for tag in verb["tags"]], srl_output["words"]
            )
            for tag, word in tags_words:
                if "ARG" in tag:
                    if tag in tag_sequences:
                        tag_sequences[tag] += f" {word}"
                    else:
                        tag_sequences[tag] = word
            for tag, sequence in tag_sequences.items():
                cleaned_sequence = remove_leading_trailing_stopwords(sequence)
                if cleaned_sequence:
                    cleaned_output.append(cleaned_sequence)
                    tag_sequences[tag] = cleaned_sequence
    # logging.warning("Not cleaning sequences.")
    return tag_sequences


def remove_leading_trailing_stopwords(string: str) -> str:
    """Returns string with leading and trailing stopwords removed.

    Args:
        nlp: A spaCy English Language model.
        string: Text to have stopwords removed.

    Returns:
        A string with no leading or trailing stopwords included.
    """
    doc = nlp(string)
    first_nonstop_idx = -1
    last_nonstop_idx = -1
    for i, token in enumerate(doc):
        if first_nonstop_idx == -1 and not token.is_stop:
            first_nonstop_idx = i
        if first_nonstop_idx > -1 and not token.is_stop:
            last_nonstop_idx = i
    clipped_doc = doc[first_nonstop_idx : last_nonstop_idx + 1]
    # Want to ignore overly long argument phrases (although probably uncommon)
    if len(clipped_doc) > 4:
        return ""
    return str(clipped_doc)


class AIDAData:
    def __init__(self) -> None:
        self.templates = self.load_templates()

    def load_templates(self):
        templates = []
        with open("./topic_list.txt", "r") as handle:
            for i, line in enumerate(handle):
                if i != 0:
                    split_line = line.split("\t")
                    templates.append(split_line[3].replace("\n", ""))
        return templates


class BaseMatcher:
    def _batch_convert_to_spacy(self, corpus: Path) -> Sequence[Doc]:
        all_docs = []
        for _file in corpus.glob("*.txt"):
            with open(_file, "r") as handle:
                doc_text = handle.read()
                parsed_english = nlp(doc_text)
                all_docs.append(parsed_english)
        return all_docs

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


class SBERTMatcher(BaseMatcher):
    def __init__(self) -> None:
        self.ss_model = SentenceTransformer("")

    def find_matches(self, corpus: List[Path]):
        pass

    def save(self, out_file: Path) -> Path:
        pass


class RegexMatcher(BaseMatcher, Matcher):
    def __init__(self) -> None:
        self.matches = None
        Matcher.__init__(self, nlp.vocab, validate=True)

    def add_patterns(self, patterns: Optional[Sequence[REGEX_PATTERN]] = None):
        for pattern, regexes in patterns.items():
            for regex in regexes:
                for row in regex:
                    possible_corona_name = row.get("TEXT")
                    if isinstance(possible_corona_name, dict):
                        possible_corona_name = possible_corona_name.get("IN")
                    if possible_corona_name == "CORONA_NAME_VARIATIONS":
                        row["TEXT"]["IN"] = CORONA_NAME_VARIATIONS
            self.add(pattern, regexes)

    def _find_links(self, sentence, tokens):
        return [disambiguate_kgtk(sentence.text, token, thresh=0.0) for token in tokens]

    def _label_roles(self, span_text: str) -> Mapping[str, str]:
        roles = srl.predict(span_text)
        if len(roles["verbs"]) > 1:
            logging.warning("More than one main verb in instance: %s", span_text)
        return clean_srl_output(roles)

    def _identify_claimer(self, span):
        for tok in span:
            if tok.pos_ == "NOUN":
                return tok.text

    def find_matches(self, corpus: Path) -> Mapping[str, str]:
        all_docs = self._batch_convert_to_spacy(corpus)

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
                claimer = self._identify_claimer(span.sent)

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

        self.matches = all_matches
        return all_matches

    def save(self, out_file: Path) -> None:
        with open(out_file, "w+") as handle:
            writer = csv.writer(handle)
            count = 0
            for match in self.matches:
                if count == 0:
                    header = match.keys()
                    writer.writerow(header)
                    count += 1
                writer.writerow(match.values())


def main():
    # Seems like we want to have higher recall (according to Daisy)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", help="Input docs", type=Path)
    p.add_argument(
        "--patterns", help="Patterns for RegexMatcher", type=Path, default=None
    )
    p.add_argument("--out", help="Out file", type=Path)
    args = p.parse_args()

    if args.patterns:
        with open(args.patterns, "r") as handle:
            patterns = json.load(handle)

    matcher = RegexMatcher()
    matcher.add_patterns(patterns)
    matcher.find_matches(args.input)
    matcher.save(args.out)


if __name__ == "__main__":
    main()
