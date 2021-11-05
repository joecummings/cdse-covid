"""
Takes the corpus files and creates AMR graphs for each sentence.

You will need to run this in your transition-amr virtual environment.
"""
import argparse
from dataclasses import dataclass
from os import getcwd, makedirs, chdir
from pathlib import Path
from typing import List, Tuple, Union, Dict, Optional

import spacy
from amr_utils.alignments import AMR_Alignment
from amr_utils.amr import AMR
from amr_utils.amr_readers import AMR_Reader
from transition_amr_parser.parse import AMRParser  # pylint: disable=import-error

from cdse_covid.semantic_extraction.run_amr_parsing import tokenize_sentence

ALIGNMENTS_TYPE = Dict[Union[List[str], str], Union[List[AMR_Alignment], list]]
AMR_READER = AMR_Reader()

@dataclass
class LocalAMR:
    doc_id: str
    graph: Tuple[List[AMR], Optional[ALIGNMENTS_TYPE]]


def tokenize_sentences(
        corpus_file: Path, spacy_tokenizer
) -> Tuple[List[str], List[List[str]]]:
    tokenized_sentences = []
    doc_sentences_to_include = []
    if corpus_file.suffix == ".txt":
        with open(corpus_file, 'r') as infile:
            input_sentences = infile.readlines()
        for sentence in input_sentences:
            tokenized_sentence = tokenize_sentence(sentence, spacy_tokenizer)
            # Blank line filter
            if len(tokenized_sentence) >= 1:
                tokenized_sentences.append(tokenized_sentence)
                doc_sentences_to_include.append(sentence)
    return doc_sentences_to_include, tokenized_sentences


def main(corpus_dir, output_dir, spacy_model, parser_path):
    cdse_path = getcwd()

    # We assume that the checkpoint is in this location within the repo
    in_checkpoint = f"{parser_path}/DATA/AMR2.0/models" \
                    "/exp_cofill_o8.3_act-states_RoBERTa-large-top24" \
                    "/_act-pos-grh_vmask1_shiftpos1_ptr-lay6-h1_grh-lay123-h2-allprev" \
                    "_1in1out_cam-layall-h2-abuf/ep120-seed42/checkpoint_best.pt"

    if not Path(in_checkpoint).exists():
        raise RuntimeError(f"Could not find checkpoint file {in_checkpoint}!")
    if not corpus_dir.exists():
        raise RuntimeError(f"Input directory {corpus_dir} could not be found!")
    if not Path(output_dir).exists():
        makedirs(output_dir)

    chdir(parser_path)
    amr_parser = AMRParser.from_checkpoint(in_checkpoint)
    chdir(cdse_path)

    for input_file in corpus_dir.iterdir():
        original_sentences, tokenized_sentences = tokenize_sentences(input_file, spacy_model.tokenizer)
        try:
            annotations = amr_parser.parse_sentences(tokenized_sentences)
        except IndexError as e:
            print(e)
            continue
        output_file = f"{output_dir}/{input_file.stem}.amr"
        with open(output_file, 'w+', encoding="utf-8") as outfile:
            # PENMAN notation is at index 0
            for anno_number, annotation in enumerate(annotations[0]):
                # Append an id to each graph so that
                # it can be loaded by amr-utils later
                outfile.write(f"# ::id {input_file.stem}_{anno_number}\n")
                outfile.write(f"# ::snt {original_sentences[anno_number]}")
                outfile.writelines(''.join(annotation))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", help="Input docs", type=Path)
    parser.add_argument("--output", help="AMR documents output dir", type=Path)
    parser.add_argument("--amr-parser-model", type=Path)

    args = parser.parse_args()

    model = spacy.load("en_core_web_sm")

    main(args.corpus, args.output, spacy_model=model, parser_path=args.amr_parser_model)
