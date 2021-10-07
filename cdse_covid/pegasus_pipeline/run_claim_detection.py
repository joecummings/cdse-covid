import json
import spacy

from vistautils.parameters_only_entrypoint import parameters_only_entry_point
from cdse_covid.claim_detection.models import RegexClaimDetector
from dataset import AIDADataset


def main(params):
    model = RegexClaimDetector()
    patterns = params.existing_file("patterns")
    with open(patterns, "r") as handle:
        patterns_json = json.load(handle)
    model.add_patterns(patterns_json)

    corpus = params.existing_directory("corpus")
    dataset = AIDADataset.from_text_files(corpus, nlp=spacy.load("en_core_web_sm"))
    matches = model.find_matches(dataset)

    out_file = params.creatable_file("out_file")
    model.save(out_file, matches)


if __name__ == "__main__":
    parameters_only_entry_point(main)
