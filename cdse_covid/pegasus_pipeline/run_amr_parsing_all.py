"""Takes the corpus files and creates AMR graphs for each sentence.

You will need to run this in your transition-amr virtual environment.
"""
import argparse
from pathlib import Path
import re
from typing import Any, List, Tuple

from amr_utils.amr_readers import AMR_Reader  # pylint: disable=import-error
import spacy
from spacy.language import Language

from cdse_covid.semantic_extraction.models.amr import AMRModel

AMR_READER = AMR_Reader()
STOP_PUNCTUATION = "!?.:;—,"


def tokenize_sentences(
    corpus_file: Path, max_tokens: int, spacy_tokenizer: Any
) -> Tuple[List[str], List[List[str]]]:
    """Tokenize multiple sentences from documents."""
    tokenized_sentences = []
    doc_sentences_to_include = []
    if corpus_file.suffix == ".txt":
        with open(corpus_file, "r", encoding="utf-8") as infile:
            input_sentences = infile.readlines()
        for sentence in input_sentences:
            tokenized_sentence = tokenize_sentence(sentence, spacy_tokenizer, max_tokens)
            # Filter for blank lines
            if len(tokenized_sentence) >= 1:
                tokenized_sentences.append(tokenized_sentence)
                doc_sentences_to_include.append(sentence)
    return doc_sentences_to_include, tokenized_sentences


def tokenize_sentence(text: str, spacy_tokenizer: Any, max_tokens: int) -> List[str]:
    """Tokenize a single sentence using provided SpaCy tokenizer."""
    tokens = spacy_tokenizer(text.strip())
    tokenized_sentence = [token.text for token in tokens]
    return refine_sentence(tokenized_sentence, max_tokens)


def refine_sentence(tokenized_sentence: List[str], max_tokens: int) -> List[str]:
    """Refine a tokenized sentence.

    If a sentence exceeds the token limit, split the sentence into clauses
    based on punctuation and keep all tokens within a clause that passes
    the threshold.

    Additionally, take any token with a format like
    "X)Y" and separate it ("X", ")", "Y") to avoid parser errors.
    """
    refined_sentence = []
    for idx, token in enumerate(tokenized_sentence):
        refined_sentence.extend(re.split(r"([()\"\[\]—])", token))
        if token in STOP_PUNCTUATION and idx >= max_tokens:
            break

    return refined_sentence


def load_amr_from_text_file(amr_file: Path, output_alignments: bool = False) -> Any:
    """Reads a document of AMR graphs and returns an AMR graph.

    If `output_alignments` is True, it will also return
    the alignment data of that graph.
    """
    if output_alignments:
        return AMR_READER.load(amr_file, remove_wiki=True, output_alignments=True)
    return AMR_READER.load(amr_file, remove_wiki=True)


def main(
    corpus_dir: Path, output_dir: Path, max_tokens: int, spacy_model: Language, parser_path: Path
) -> None:
    """Entrypoint to AMR parsing over entire document."""
    amr_parser = AMRModel.from_folder(parser_path)

    for input_file in corpus_dir.iterdir():
        original_sentences, tokenized_sentences = tokenize_sentences(
            input_file, max_tokens, spacy_model.tokenizer
        )
        # Attempting to AMR-parse an empty list yields an error
        if not tokenized_sentences:
            continue
        amr_parse = amr_parser.amr_parse_sentences(tokenized_sentences)

        output_file = f"{output_dir}/{input_file.stem}.amr"
        with open(output_file, "w+", encoding="utf-8") as outfile:
            # PENMAN notation is at index 0
            for anno_number, annotation in enumerate(amr_parse.annotations[0]):
                # Append an id to each graph so that
                # it can be loaded by amr-utils later
                outfile.write(f"# ::id {input_file.stem}_{anno_number}\n")
                outfile.write(f"# ::snt {original_sentences[anno_number]}")
                outfile.writelines("".join(annotation))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", help="Input docs", type=Path)
    parser.add_argument("--output", help="AMR documents output dir", type=Path)
    parser.add_argument("--amr-parser-model", type=Path)
    parser.add_argument(
        "--max-tokens", help="Max tokens allowed in a sentence to be parsed", type=int, default=50
    )

    args = parser.parse_args()

    model = spacy.load("en_core_web_sm")

    main(
        args.corpus,
        args.output,
        args.max_tokens,
        spacy_model=model,
        parser_path=args.amr_parser_model,
    )
