from dataclasses import dataclass
from os import chdir, getcwd
from pathlib import Path
from typing import Any, List
import uuid

from transition_amr_parser.parse import AMRParser  # pylint: disable=import-error

from amr_utils.amr_readers import AMR_Reader, Metadata_Parser


@dataclass
class AMROutput:
    label_id: int
    graph: Any
    alignments: Any
    annotations: Any


class AMRModel(object):

    def __init__(self, parser) -> None:
        self.parser = parser

    @classmethod
    def from_folder(cls, folder: Path) -> "AMRModel":
        cdse_path = getcwd()
        # We assume that the checkpoint is in this location within the repo
        in_checkpoint = f"{folder}/DATA/AMR2.0/models" \
                        "/exp_cofill_o8.3_act-states_RoBERTa-large-top24" \
                        "/_act-pos-grh_vmask1_shiftpos1_ptr-lay6-h1_grh-lay123-h2-allprev" \
                        "_1in1out_cam-layall-h2-abuf/ep120-seed42/checkpoint_best.pt"
        chdir(folder)
        parser = AMRParser.from_checkpoint(in_checkpoint)
        chdir(cdse_path)
        return cls(parser)
    
    def amr_parse_sentences(self, sentences: List[List[str]], output_alignments=False) -> AMROutput:
        annotations = self.parser.parse_sentences(sentences)
        metadata, graph_metadata = Metadata_Parser().readlines(annotations[0][0])
        amr, alignments = AMR_Reader._parse_amr_from_metadata(metadata["tok"], graph_metadata)
        return AMROutput(int(uuid.uuid1()), amr, alignments, annotations)