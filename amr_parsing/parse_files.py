"""
Takes the corpus files and creates AMR graphs for each sentence.

You will need to run this in your transition-amr virtual environment.
"""
import argparse
from os import chdir, getcwd, makedirs
from pathlib import Path
from typing import List, Tuple

import spacy
from transition_amr_parser.parse import AMRParser

NLP = spacy.load("en_core_web_sm")
TOKENIZER = NLP.tokenizer


def main(arg_parser):
    arg_parser.add_argument("--input", help="Input docs", type=Path)
    arg_parser.add_argument("--out", help="AMR output dir", type=Path)

    args = arg_parser.parse_args()
    parser_path = "../transition-amr-parser"
    cdse_path = getcwd()

    # We assume that the checkpoint is in this location within the repo
    in_checkpoint = f"{parser_path}/DATA/AMR2.0/models" \
                    "/exp_cofill_o8.3_act-states_RoBERTa-large-top24" \
                    "/_act-pos-grh_vmask1_shiftpos1_ptr-lay6-h1_grh-lay123-h2-allprev" \
                    "_1in1out_cam-layall-h2-abuf/ep120-seed42/checkpoint_best.pt"

    if not Path(in_checkpoint).exists():
        raise RuntimeError(f"Could not find checkpoint file {in_checkpoint}!")
    if not args.input.exists():
        raise RuntimeError(f"Input directory {args.input} could not be found!")
    if not Path(args.out).exists():
        makedirs(args.out)

    chdir(parser_path)
    amr_parser = AMRParser.from_checkpoint(in_checkpoint)
    chdir(cdse_path)
    for input_file in args.input.iterdir():
        doc_sentences, tokenized_sentences = tokenize_sentences(input_file)
        annotations = amr_parser.parse_sentences(tokenized_sentences)

        output_file = f"{args.out}/{input_file.stem}.amr"
        with open(output_file, 'w+', encoding="utf-8") as outfile:
            # PENMAN notation is at index 0
            for anno_number, annotation in enumerate(annotations[0]):
                # Append an id to each graph so that
                # it can be loaded by amr-utils later
                outfile.write(f"# ::id {input_file.stem}_{anno_number}\n")
                outfile.write(f"# ::snt {doc_sentences[anno_number]}")
                outfile.writelines(''.join(annotation))
        print(f"AMR output saved to {Path(output_file).absolute()}")


def tokenize_sentences(corpus_file) -> Tuple[List[str], List[List[str]]]:
    tokenized_sentences = []
    final_doc_sentences = []
    with open(corpus_file, 'r') as infile:
        input_sentences = infile.readlines()
    for sentence in input_sentences:
        tokens = TOKENIZER(sentence.strip())
        tokenized_sentence = [token.text for token in tokens]
        if len(tokenized_sentence) >= 1:
            tokenized_sentences.append(tokenized_sentence)
            final_doc_sentences.append(sentence)
    return final_doc_sentences, tokenized_sentences


if __name__ == "__main__":
    main(argparse.ArgumentParser())
