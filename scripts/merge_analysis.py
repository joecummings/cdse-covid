import argparse
from collections import defaultdict
from pathlib import Path
from rdflib import Graph
from cdse_covid.pegasus_pipeline.merge.merge_aif import get_claims, get_claim_semantics_for_claim
import pandas as pd
from tqdm import tqdm


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--merged-aif-folder", help="Path to merged AIF folder.", type=Path)
    p.add_argument("--output", help="Path to output CSV.", type=Path)
    args = p.parse_args()

    added_claim_info = defaultdict(lambda: defaultdict(list))

    for aif in tqdm(args.merged_aif_folder.glob("*.ttl"), total=1180):
        g = Graph()
        g = g.parse(source=aif, format="turtle")

        all_claims = get_claims(g)

        for claim in all_claims:
            cs = get_claim_semantics_for_claim(g, claim[0])

            for elem in cs:
                elem = elem[0]
                if elem.find("cdse") != -1:
                    if elem.find("events") != -1:
                        added_claim_info[claim[0]]["events"].append(elem)
                    elif elem.find("entity") != -1:
                        added_claim_info[claim[0]]["entities"].append(elem)

    print(f"Number of claims modified: {len(added_claim_info)}.")
    print(f'Number of events added: {sum(len(c["events"]) for _, c in added_claim_info.items())}')
    print(
        f'Number of entities added: {sum(len(c["entities"]) for _, c in added_claim_info.items())}'
    )

    df = pd.DataFrame.from_dict(added_claim_info, orient="index")
    df.to_csv(args.output)


if __name__ == "__main__":
    main()
