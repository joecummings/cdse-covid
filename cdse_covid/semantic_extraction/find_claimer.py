import argparse
from pathlib import Path

from amr_utils.amr import AMR
from cdse_covid.claim_detection.models import Claim
from cdse_covid.claim_detection.run_claim_detection import ClaimDataset
from cdse_covid.semantic_extraction.entities import Claimer
from cdse_covid.semantic_extraction.models.amr import AMRFinder, AMRModel
from cdse_covid.semantic_extraction.utils.amr_extraction_utils import \
    ALIGNMENTS_TYPE
from cdse_covid.semantic_extraction.utils.claimer_utils import identify_claimer


class AMRClaimerFinder(AMRFinder):
    def __init__(self, *, model: AMRModel = None, amr_graph: AMR = None, amr_alignments: ALIGNMENTS_TYPE = None) -> None:
        super().__init__(model=model, amr_graph=amr_graph, amr_alignments=amr_alignments)

    def find_claimer(self, claim: Claim) -> Claimer:
        claimer = identify_claimer(claim.claim_sentence, self.amr_graph, self.amr_alignments)
        return claimer


def main(input, output, amr_model, amr_documents):
    claims = ClaimDataset.load_from_dir(input)

    if amr_documents:
        finder = AMRClaimerFinder.from_amr_file(amr_documents)
    else:
        finder = AMRClaimerFinder.from_amr_model(amr_model)
    
    for claim in claims:
        claimer = finder.find_claimer(claim)
        claim.claimer = claimer
    
    claims.save_to_dir(output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-claims", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--amr-model", type=Path, default=None)
    parser.add_argument("--amr-documents", type=Path, default=None)
    args = parser.parse_args()

    main(
        args.input_claims,
        args.output,
        args.amr_model, 
        args.amr_documents
    )
