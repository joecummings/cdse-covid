import json

from vistautils.parameters_only_entrypoint import parameters_only_entry_point
from cdse_covid.claim_detection.models import RegexClaimDetector


def main(params):
    model = RegexClaimDetector()
    patterns = params.existing_file("patterns")
    with open(patterns, "r") as handle:
        patterns_json = json.load(handle)
    model.add_patterns(patterns_json)

    corpus = params.existing_directory("corpus")
    matches = model.find_matches(corpus)

    out_file = params.creatable_file("out_file")
    model.save(out_file, matches)


if __name__ == "__main__":
    parameters_only_entry_point(main)
