from pathlib import Path
from typing import List

from amr_utils.amr import AMR
from cdse_covid.semantic_extraction.utils.amr_extraction_utils import ALIGNMENTS_TYPE, load_amr_from_text_file

from transition_amr_parser.parse import AMRParser  # pylint: disable=import-error

from amr_utils.amr_readers import AMR_Reader, Matedata_Parser


class AMRModel:

    def __init__(self, parser) -> None:
        self.parser = parser

    @classmethod
    def from_folder(cls, folder: Path) -> "AMRModel":
        # We assume that the checkpoint is in this location within the repo
        in_checkpoint = f"{folder}/DATA/AMR2.0/models" \
                        "/exp_cofill_o8.3_act-states_RoBERTa-large-top24" \
                        "/_act-pos-grh_vmask1_shiftpos1_ptr-lay6-h1_grh-lay123-h2-allprev" \
                        "_1in1out_cam-layall-h2-abuf/ep120-seed42/checkpoint_best.pt"
        parser = AMRParser.from_checkpoint(in_checkpoint)
        return cls(parser)
    
    def amr_parse_sentences(self, sentences: List[str], output_alignments=False):
        annotations = self.parser.parse_sentences(sentences)
        metadata, graph_metadata = Matedata_Parser().readlines(annotations[0][0])
        amr, alignments = AMR_Reader._parse_amr_from_metadata(metadata["tok"], graph_metadata)
        return amr, alignments


class AMRFinder:

    def __init__(self, *, model: AMRModel = None, amr_graph: AMR = None, amr_alignments: ALIGNMENTS_TYPE = None) -> None:
        self.model = model
        self.amr_graph = amr_graph
        self.amr_alignments = amr_alignments

    @classmethod
    def from_amr_file(cls, file: Path) -> "AMRFinder":
        amr = load_amr_from_text_file(file, output_alignments=True)
        return cls(amr_graph=amr.graph, amr_alignments=amr.alignments)
    
    @classmethod
    def from_amr_model(cls, path_to_model: Path) -> "AMRFinder":
        model = AMRModel.from_folder(path_to_model)
        return cls(model=model)