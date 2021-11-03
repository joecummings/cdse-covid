"""Run claim detection over corpus of SpaCy encoded documents."""
import argparse
import logging
from pathlib import Path
import spacy
from spacy.language import Language
from cdse_covid.dataset import AIDADataset
import csv

from cdse_covid.claim_detection.amr_claim_detector import AMRClaimDetector
from cdse_covid.claim_detection.regex_claim_detector import RegexClaimDetector



def main(input_corpus: Path, out_dir: Path, spacy_model: Language, * , patterns: Path, amr_parser_path: Path):

    # regex_model = RegexClaimDetector(spacy_model)
    # with open(patterns, "r") as handle:
    #     patterns_json = json.load(handle)
    # regex_model.add_patterns(patterns_json)

    topics_info = []
    with open(Path(__file__).parent / "topic_list.txt", "r") as handle:
        reader = csv.reader(handle, delimiter="\t")
        headers = next(reader)
        for row in reader:
            topics_info.append({item: header for item, header in zip(row, headers)})

    dataset = AIDADataset.from_serialized_docs(input_corpus)
    
    # matches = regex_model.generate_candidates(dataset, spacy_model.vocab)

    amr_model = AMRClaimDetector(amr_parser_path, topics_info)
    matches = amr_model.generate_candidates(dataset)

    for claim in matches:
        template = topics_info[claim.claim_template]
        claim.topic = template["topic"]
        claim.subtopic = template["subtopic"]
                
    matches.save_to_dir(out_dir)

    logging.info("Saved matches to %s", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Input docs", type=Path)
    parser.add_argument(
        "--patterns", help="Patterns for Regex", type=Path, default=None
    )
    parser.add_argument("--amr-parser", help="Path to AMR Parser if using that type of claim detection.", type=Path, default=None)
    parser.add_argument("--out", help="Out file", type=Path)
    parser.add_argument("--spacy-model", type=Path)
    args = parser.parse_args()

    model = spacy.load("en_core_web_sm")

    main(args.input, args.out, model, patterns=args.patterns, amr_parser_path=args.amr_parser)
