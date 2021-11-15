"""Analyze the claims and semantics being produced."""
import argparse
from collections import defaultdict
import csv
import json
from typing import Any, MutableMapping, Set


def main() -> None:
    """Entrypoint into analysis."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", help="JSON input file")
    p.add_argument(
        "--wikidata",
        action="store_true",
        help="Evaluate accuracy of WikiData Qnodes.",
        default=False,
    )
    args = p.parse_args()

    with open(args.input, "r", encoding="utf-8") as handle:
        claims = json.load(handle)

    # distributions
    num_of_each_topic: MutableMapping[str, int] = defaultdict(int)
    num_of_each_subtopic: MutableMapping[str, int] = defaultdict(int)

    for claim in claims:
        num_of_each_topic[claim["topic"]] += 1
        num_of_each_subtopic[claim["subtopic"]] += 1

    with open("topic_distribution.csv", "w+", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for k, v in num_of_each_topic.items():
            writer.writerow([k, v])

    with open("subtopic_distribution.csv", "w+", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for k, v in num_of_each_subtopic.items():
            writer.writerow([k, v])

    # claim detection accuracy
    num_of_claims = len(claims)
    print(f"Num of claims: {num_of_claims}")

    # x variable & qnode accuracy
    num_x_variable = 0
    num_x_q_nodes = 0
    for claim in claims:
        if claim["x_variable"]:
            num_x_variable += 1
        if claim["x_variable_qnode"]:
            num_x_q_nodes += 1
    print(f"% X variables found: {num_x_variable / num_of_claims}")
    print(f"% X variable qnodes found: {num_x_q_nodes / num_x_variable}")

    # claimer & qnode accuracy
    num_claimers = 0
    num_claimer_q_nodes = 0
    for claim in claims:
        if claim["claimer"]:
            num_claimers += 1
        if claim["claimer_qnode"]:
            num_claimer_q_nodes += 1
    print(f"% of claimers found over total: {num_claimers / num_of_claims}")
    print(f"% of claimer qnodes found: {num_claimer_q_nodes / num_claimers}")

    # overall wikidata accuracy
    if args.wikidata:
        prompt = "Is this QNODE_TYPE qnode appropriate for the sentence above? (1=yes, 0=no):"
        valid_res = {1, 0}
        wikidata_total = 0
        wikidata_good = 0
        for claim in claims:
            print(claim["claim_sentence"])
            if claim["x_variable_qnode"]:
                good = evaluate_appropriate_qnode(
                    claim["x_variable_qnode"], prompt, valid_res, "X Variable"
                )
                wikidata_good += good
                wikidata_total += 1
            if claim["claimer_qnode"]:
                good = evaluate_appropriate_qnode(
                    claim["claimer_qnode"], prompt, valid_res, "claimer"
                )
                wikidata_good += good
                wikidata_total += 1
            if claim["claim_semantics"]:
                if claim["claim_semantics"]["event"]:
                    good = evaluate_appropriate_qnode(
                        claim["claim_semantics"]["event"], prompt, valid_res, "event"
                    )
                    wikidata_good += good
                    wikidata_total += 1
                if claim["claim_semantics"]["args"]:
                    for role, arg in claim["claim_semantics"]["args"].items():
                        good = evaluate_appropriate_qnode(arg, prompt, valid_res, role)
                        wikidata_good += good
                        wikidata_total += 1
        print(f"% of accurate qnode selections: {wikidata_good / wikidata_total}")


def evaluate_appropriate_qnode(
    qnode: MutableMapping[str, Any], prompt: str, valid_res: Set[int], type_of_qnode: str
) -> int:
    """Loop to evaluate if qnode is appropriate."""
    good = -1
    prompt = prompt.replace("QNODE_TYPE", type_of_qnode)
    while good not in valid_res:
        good = int(input(f"{prompt}\n{qnode}\n"))
        if good not in valid_res:
            print(f"Invalid input: {good}. Please only input a 1 for yes or a 0 for no.\n")
    return good


if __name__ == "__main__":
    main()
