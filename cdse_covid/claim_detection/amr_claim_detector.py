
from pathlib import Path
from typing import List, Mapping, Any
from cdse_covid.claim_detection.models import ClaimDataset
from cdse_covid.claim_detection.regex_claim_detector import ClaimDetector

from cdse_covid.dataset import AIDADataset
from transition_amr_parser.parse import AMRParser # pylint: disable=import-error

from amr_utils.amr_readers import AMR_Reader, Matedata_Parser


class AMRClaimDetector(ClaimDetector):

    def __init__(self, amr_parser_path: Path, topics_info: List[Mapping[str, Any]] = None) -> None:
        self.parser_path = amr_parser_path
        self.topics_info = topics_info
    
    def generate_candidates(self, corpus: AIDADataset):
        claim_dataset = ClaimDataset()

        # We assume that the checkpoint is in this location within the repo
        in_checkpoint = f"{self.parser_path}/DATA/AMR2.0/models" \
                        "/exp_cofill_o8.3_act-states_RoBERTa-large-top24" \
                        "/_act-pos-grh_vmask1_shiftpos1_ptr-lay6-h1_grh-lay123-h2-allprev" \
                        "_1in1out_cam-layall-h2-abuf/ep120-seed42/checkpoint_best.pt"

        if not Path(in_checkpoint).exists():
            raise RuntimeError(f"Could not find checkpoint file {in_checkpoint}!")

        amr_parser = AMRParser.from_checkpoint(in_checkpoint)
            
        for doc in corpus.documents:
            annotations = amr_parser.parse_sentences(doc.sents)
            metadata, graph_metadata = Matedata_Parser().readlines(annotations[0][0])
            amr, alignments = AMR_Reader._parse_amr_from_metadata(metadata["tok"], graph_metadata)

