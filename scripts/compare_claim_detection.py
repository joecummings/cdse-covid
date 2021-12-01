"""Script to compare claims produced by UIUC and claims produced by ISI."""
import argparse
import json
from pathlib import Path


def main() -> None:
    """Running comparison of claims."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--uiuc-claims", help="Path to UIUC claims in JSON format.", type=Path)
    p.add_argument("--isi-claims", help="Path to ISI claims in JSON format.", type=Path)
    args = p.parse_args()

    # UIUC JSON is split by Doc ID
    with open(args.uiuc_claims, "r", encoding="utf-8") as handle:
        uiuc = json.load(handle)

    num_uiuc_claims = sum(len(v) for v in uiuc.values())
    print(f"# of UIUC detected claims: {num_uiuc_claims}")

    with open(args.isi_claims, "r", encoding="utf-8") as handle:
        isi = json.load(handle)

    num_isi_claims = len(isi)
    print(f"# of ISI detected claims: {num_isi_claims}")

    overlap = 0
    fuzzy_overlap = 0
    for claim in isi:
        doc_id = claim["doc_id"].replace(".rsd", "")  # Hack to remove the RSD file ending
        uiuc_claims_for_doc = uiuc[doc_id]
        for uiuc_claim in uiuc_claims_for_doc:
            span_start = uiuc_claim["claim_span_start"]
            span_end = uiuc_claim["claim_span_end"]

            if claim["claim_span"] == [span_start, span_end]:
                overlap += 1

            if claim["claim_sentence"] == uiuc_claim["sentence"]:
                fuzzy_overlap += 1

    print(f"# of claims detected by both UIUC & ISI: {fuzzy_overlap}")


if __name__ == "__main__":
    main()
