"""
Takes the corpus files and creates AMR graphs for each sentence.

You will need to run this in your transition-amr virtual environment.
"""
from os import chdir, getcwd, makedirs
import sys
from pathlib import Path
from typing import List

import spacy
from transition_amr_parser.parse import AMRParser

NLP = spacy.load("en_core_web_sm")
TOKENIZER = NLP.tokenizer


def main(args):
    transition_amr_repo_path = args[1]
    input_dir = Path(args[2])
    output_dir = args[3]

    cdse_path = getcwd()

    # We assume that the checkpoint is in this location within the repo
    in_checkpoint = f"{transition_amr_repo_path}/DATA/AMR2.0/models" \
                    "/exp_cofill_o8.3_act-states_RoBERTa-large-top24" \
                    "/_act-pos-grh_vmask1_shiftpos1_ptr-lay6-h1_grh-lay123-h2-allprev" \
                    "_1in1out_cam-layall-h2-abuf/ep120-seed42/checkpoint_best.pt"

    if not Path(in_checkpoint).exists():
        raise RuntimeError(f"Could not find checkpoint file {in_checkpoint}!")
    if not input_dir.exists():
        raise RuntimeError(f"Input directory {input_dir} could not be found!")
    if not Path(output_dir).exists():
        makedirs(output_dir)

    chdir(transition_amr_repo_path)
    amr_parser = AMRParser.from_checkpoint(in_checkpoint)
    chdir(cdse_path)
    for input_file in input_dir.iterdir():
        doc_sentences = tokenize_sentences(input_file)
        annotations = amr_parser.parse_sentences(doc_sentences)

        output_file = f"{output_dir}/{input_file.stem}.amr"
        with open(output_file, 'w+', encoding="utf-8") as outfile:
            # PENMAN notation is at index 0
            for anno_number, annotation in enumerate(annotations[0]):
                # Append an id to each graph
                # so that it can be loaded by amr-utils
                outfile.write(f"# ::id {input_file.stem}_{anno_number}\n")
                outfile.writelines(''.join(annotation))
        print(f"AMR output saved to {Path(output_file).absolute()}")


def tokenize_sentences(corpus_file) -> List[List[str]]:
    tokenized_sentences = []
    with open(corpus_file, 'r') as infile:
        input_sentences = infile.readlines()
    for sentence in input_sentences:
        tokens = TOKENIZER(sentence)
        tokenized_sentences.append([token.text for token in tokens])
    return tokenized_sentences


if __name__ == "__main__":
    main(sys.argv)
