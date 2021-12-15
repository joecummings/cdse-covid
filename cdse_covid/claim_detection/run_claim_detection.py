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
from nltk import ngrams

import spacy
from spacy.language import Language
from spacy.matcher import Matcher
from spacy.tokens import Span
from torch.functional import Tensor

from cdse_covid.claim_detection.claim import TOKEN_OFFSET_THEORY, Claim, create_id
from cdse_covid.dataset import AIDADataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import pytorch_cos_sim, pairwise_cos_sim
import tqdm
import numpy as np
import stanza
from stanza.server import CoreNLPClient

CORONA_NAME_VARIATIONS = [
    "COVID-19",
    "Coronavirus",
    "SARS-CoV-2",
    "coronavirus",
    "corona virus",
    "covid-19",
    "covid 19",
]

nlp = stanza.Pipeline("en", processors="tokenize,pos,constituency")

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


class SBERTClaimDetector(ClaimDetector):
    """Detect claims using SentenceBERT encoded similarity."""

    def __init__(self, sbert_model: SentenceTransformer) -> None:
        self.sbert_model = sbert_model

    def encode_topics(self, topics: List[str]) -> List[Tensor]:
        return self.sbert_model.encode(topics, convert_to_tensor=True)

    def get_noun_phrases(self, text):
        # English example
        with CoreNLPClient(timeout=30000, memory="16G", endpoint="http://localhost:9000") as client:
            return self.noun_phrases(client, text, _annotators="tokenize,pos,parse")

    # get noun phrases with tregex
    def noun_phrases(self, _client, _text, _annotators=None):
        pattern = "S"
        return _client.tregex(_text, pattern, annotators=_annotators)

    def generate_candidates(self, corpus: AIDADataset, *args: Any) -> ClaimDataset:
        encoded_topics = args[0]
        topics = list(args[1])

        claims = ClaimDataset()
        # parser = benepar.Parser("benepar_en3")

        for doc in tqdm.tqdm(corpus):
            raw_sent = doc.strip("\n")
            cleaned_sents = []
            cleaned_ngrams = []
            chunk_to_sent = {}
            idx_of_chunk = 0
            for i, sent in enumerate([raw_sent]):
                for name in CORONA_NAME_VARIATIONS:
                    sent = sent.replace(name, "COVID-19")
                cleaned_sents.append(sent)

                nps = self.get_noun_phrases(sent)

                if nps:
                    for match in nps["sentences"][0].values():
                        if len(match) >= 4 and match["spanString"] not in cleaned_ngrams:
                            cleaned_ngrams.append(match["spanString"])
                            chunk_to_sent[idx_of_chunk] = i
                            idx_of_chunk += 1
                # start = 0
                # token_sent = sent.split()
                # for _ in range(len(token_sent) // 4 + 1):
                #     if start + 4 <= len(token_sent):
                #         chunk = token_sent[start:start+4]
                #         start += 4
                #     else:
                #         chunk = token_sent[start:]
                #         while len(chunk) < 4:
                #             chunk.append("[PAD]")
                #     chunk_to_sent[idx_of_chunk] = i
                #     idx_of_chunk += 1
                #     cleaned_ngrams.append(" ".join(chunk))
            # If using PhraseBERT, should use constituency parsing in order to extract all NP, VP, ADJP & ADVP
            encoded_sentences = self.sbert_model.encode(cleaned_ngrams, convert_to_tensor=True)
            sim_matches = pytorch_cos_sim(encoded_topics, encoded_sentences)
            for i, match in enumerate(sim_matches):
                to_add = filter(lambda x: x[1] > 0.50, enumerate(match))
                for idx, _ in to_add:
                    
                    sentence = cleaned_sents[chunk_to_sent[idx]]
                    start_idx = sentence.index(cleaned_ngrams[idx])
                    claim = Claim(
                        claim_id=create_id(),
                        doc_id="",
                        claim_text=cleaned_ngrams[idx],
                        claim_span=(start_idx, start_idx + len(cleaned_ngrams[idx])),
                        claim_sentence=sentence,
                        claim_template=topics[i],
                    )
                    claims.add_claim(claim)
            breakpoint()
        return claims


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

        with open("all_docs_as_sentences.csv", "w+") as handle:
            writer = csv.writer(handle)

            for doc in corpus.documents:

                for sent in doc[1].sents:
                    writer.writerow([doc[0], sent])

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
                        claim_id=create_id(),
                        doc_id=doc[0],
                        claim_text=span.text,
                        claim_sentence=span.sent.text,
                        claim_span=(span_offset_start, span_offset_end),
                        claim_template=rule_id,
                    )
                    new_claim.add_theory(TOKEN_OFFSET_THEORY, claim_sentence_tokens_to_offsets)

                    claim_dataset.add_claim(new_claim)

            return claim_dataset


def clean_templates(dirty_templates):
    cleaned_templates = []
    for sentence in dirty_templates:
        for name in CORONA_NAME_VARIATIONS:
            sentence = sentence.replace(name, "COVID-19")
        sentence = sentence.replace("X", "someone or something")
        cleaned_templates.append(sentence)
    return cleaned_templates


def main(input_corpus: Path, patterns: Path, out_dir: Path, *, spacy_model: Language) -> None:
    """Entrypoint to claim detection script."""
    # dataset = AIDADataset.from_serialized_docs(input_corpus)
    claims = []
    with open(input_corpus, "r") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line in reader:
            if line[1] == "1":
                claims.append(line[0])

    # regex_model = RegexClaimDetector(spacy_model)
    # with open(patterns, "r", encoding="utf-8") as handle:
    #     patterns_json = json.load(handle)
    # regex_model.add_patterns(patterns_json)
    # matches = regex_model.generate_candidates(dataset, spacy_model.vocab)

    # sbert_model = SentenceTransformer("all-mpnet-base-v2") #"whaleloops/phrase-bert")
    sbert_model = SentenceTransformer("whaleloops/phrase-bert")
    ss_model = SBERTClaimDetector(sbert_model)

    topics_info = {}
    with open(Path(__file__).parent / "topic_list.txt", "r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for i, row in enumerate(reader):
            if i == 0:
                continue
            key = row[3]
            topics_info[key] = {"topic": row[2], "subtopic": row[1]}

    cleaned_templates = clean_templates(list(topics_info.keys()))
    encoded_topics = ss_model.encode_topics(cleaned_templates)
    matches = ss_model.generate_candidates(claims, encoded_topics, topics_info.keys())

    with open("ss_4_chunks_01_from_claimbuster_100_phrase.csv", "w+") as handle:
        writer = csv.writer(handle)
        writer.writerow(["claim_id", "doc_id", "claim_text", "sentence", "claim_template"])
        for claim in matches:
            writer.writerow(
                [
                    claim.claim_id,
                    claim.doc_id,
                    claim.claim_text,
                    claim.claim_sentence,
                    claim.claim_template,
                ]
            )

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
