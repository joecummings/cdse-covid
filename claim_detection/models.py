import abc
import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Dict, Set

from allennlp_models.pretrained import load_predictor
from amr_utils.amr_readers import AMR_Reader
from amr_utils.amr import AMR
from amr_utils.propbank_frames import propbank_frames_dictionary
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
from spacy.matcher import Matcher
from spacy.tokens import Span
import spacy

from wikidata_linker.wikidata_linking import disambiguate_kgtk

from srl_utils import clean_srl_output
from dataset import AIDADataset

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

CLAIMER_ROLES = {
    "admitter",
    "arguer",
    "asserter",
    "claimer",
    "sayer",
    "speaker",
}

# ss_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

NLP = spacy.load("en_core_web_sm")
SRLModel = load_predictor("structured-prediction-srl")
TOKENIZER = NLP.tokenizer

AMR_READER = AMR_Reader()
LEMMATIZER = WordNetLemmatizer()

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
        self.ss_model = SentenceTransformer("")

    def find_matches(self, corpus: List[Path]):
        pass

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

    def _identify_claimer(self, span: Span, amr: Optional[AMR]) -> str:
        """Identify the claimer of the span."""
        if not amr:
            return ""
        # Find the claim node by using one of two rules:
        # 1) Search for the Statement node of the claim by working up the graph
        # 2) If (1) fails, find the first Statement node in the graph
        claim_node = get_claim_node(span.text, amr)
        claimer_list = get_argument_node(amr, claim_node)
        return ", ".join(claimer_list)

    def find_matches(self, corpus: Path) -> Mapping[str, str]:
        docs_to_amrs = AIDADataset(
            nlp=NLP, templates_file=TEMPLATE_FILE
        )._batch_convert_amr_to_spacy(corpus)
        all_docs = docs_to_amrs.keys()

        all_matches = []
        for doc in all_docs:
            matches = self.__call__(doc)
            sentences_to_amrs = {}
            doc_amrs = docs_to_amrs.get(doc)
            if doc_amrs:
                sentences_to_amrs = {
                    " ".join(amr_graph.tokens): amr_graph
                    for amr_graph in doc_amrs
                }
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
                if doc_amrs:
                    claim_amr = get_amr_for_claim(span.text, sentences_to_amrs)
                    claimer = self._identify_claimer(span, claim_amr)
                else:
                    print(
                        "No corresponding AMR graph found;"
                        " no claimer will be identified."
                    )
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


def get_amr_for_claim(span: str, sents_to_amrs: Dict[str, AMR]) -> Optional[AMR]:
    for sentence, amr in sents_to_amrs.items():
        if span in sentence:
            return amr
    return None


def get_claim_node(claim_text: str, amr: AMR) -> Optional[str]:
    """Get the head node of the claim"""
    graph_nodes = amr.nodes
    claim_tokens = claim_text.split(" ")
    for token in claim_tokens:
        for node, label in graph_nodes.items():
            # We're hoping that at least one nominal/verbial lemma is found
            if (
                    LEMMATIZER.lemmatize(token, pos="n") == label or
                    LEMMATIZER.lemmatize(token, pos="v") == label
            ):
                return get_claim_node_from_token(node, graph_nodes, amr.edges)
    return search_for_claim_node(graph_nodes)


def is_claim_frame(node_label: str) -> bool:
    if node_label in propbank_frames_dictionary:
        for claimer_role in CLAIMER_ROLES:
            if claimer_role in propbank_frames_dictionary[node_label].lower():
                return True
    return False


def get_claim_node_from_token(node, node_dict, edges) -> Optional[str]:
    """Fetch the claim node by traveling up from a child node"""
    for parent_node, _, arg_node in edges:
        if arg_node == node:
            # Check if the parent is a claim node
            parent_label = node_dict[parent_node]
            if is_claim_frame(parent_label):
                return parent_node
            else:
                return get_claim_node_from_token(parent_node, node_dict, edges)
    return search_for_claim_node(node_dict)


def search_for_claim_node(graph_nodes) -> Optional[str]:
    """Rule #2: try finding the statement by reading through all of the nodes"""
    for node, label in graph_nodes.items():
        if is_claim_frame(label):
            return node
    return None


def create_amr_dict(amr: AMR) -> Dict[str, Dict[str, str]]:
    amr_dict = defaultdict(dict)
    for parent, role, arg in amr.edges:
        amr_dict[parent][role] = arg
    return amr_dict


def get_argument_node(amr: AMR, claim_node: Optional[str]) -> List[str]:
    """Get all argument (claimer) nodes of the claim node"""
    claimers = set()
    nodes = amr.nodes
    amr_dict = create_amr_dict(amr)
    node_args = amr_dict.get(claim_node)
    if node_args:
        claimer_node = node_args.get(":ARG0")
        claimer_label = nodes[claimer_node]
        if claimer_label == "person":
            name = get_claimer_name(amr_dict, nodes, claimer_node)
            if len(name) > 0:
                claimers.add(f"'{name}'")
        elif claimer_label == "and":
            for role, arg_node in amr_dict[claimer_node].items():
                co_claimer_label = nodes[arg_node]
                if co_claimer_label == "person":
                    name = get_claimer_name(amr_dict, nodes, arg_node)
                    if len(name) > 0:
                        claimers.add(f"'{name}'")
                else:
                    claimers.add(co_claimer_label)
        else:
            claimers.add(claimer_label)
    return list(claimers)


def get_claimer_name(
        amr_dict: Dict[str, Dict[str, str]],
        node_dict: Dict[str, str],
        person_node: str
) -> str:
    name_strings = []
    name_node = amr_dict[person_node].get(":name")
    if name_node:
        person_args = amr_dict[name_node]
        for role, node in person_args.items():
            if role.startswith(":op"):
                name_strings.append(node_dict[node].strip("\""))
    return " ".join(name_strings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Input AMR docs", type=Path)
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


if __name__ == "__main__":
    main()
