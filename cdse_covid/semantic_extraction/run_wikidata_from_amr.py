"""Runs Wikidata linking over selection of AMR graphs."""
import argparse
from pathlib import Path
from cdse_covid.claim_detection.run_claim_detection import ClaimDataset
from cdse_covid.semantic_extraction.amr_extraction_utils import load_amr_from_text_file
from cdse_covid.semantic_extraction.entities import AMRLabel
from wikidata_linker.disambiguate_with_amr_v2 import disambiguate_with_amr_v2
from cdse_covid.semantic_extraction.run_wikidata_linking import WikidataQnode

def main(claims_input: Path, output: Path):
    claims = ClaimDataset.load_from_dir(claims_input)

    for claim in claims:
        amr_theory: AMRLabel = claim.get_theory("amr")
        qnode = disambiguate_with_amr_v2(claim.doc_id, 0, amr_theory.graph)
        if claim.doc_id == qnode.doc_id:
            if not claim.claim_semantics:
                claim.claim_semantics = {}
            if qnode.event_qnode:
                claim.claim_semantics["event"] = {
                    "name": qnode.event_qnode["pb"],
                    "qnode": WikidataQnode(
                        qnode_id=qnode.event_qnode["qnode"],
                        label=qnode.event_qnode["name"],
                        description=qnode.event_qnode["definition"],
                    )
                }
            if qnode.args_qnodes:
                claim.claim_semantics["args"] = {}
                for role, arg in qnode.args_qnodes.items():
                    if arg.get("pagerank"):
                        name = arg["label"][0]
                        description = arg["description"][0]
                    else:
                        name = arg["rawName"]
                        description = arg["definition"]
                    claim.claim_semantics["args"][role] = {
                        "name": name,
                        "qnode": WikidataQnode(
                            qnode_id=arg["qnode"],
                            label=name,
                            description=description
                        )
                    }
    claims.save_to_dir(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-input", type=Path)
    parser.add_argument("--output", type=Path)

    args = parser.parse_args()

    main(
        args.claim_input,
        args.output,
    )