"""Classes related to AMR Model."""
from dataclasses import dataclass
import logging
from os import chdir, getcwd
from pathlib import Path
from typing import Any, List
import uuid

from amr_utils.amr_readers import AMR_Reader, Metadata_Parser  # pylint: disable=import-error
from transition_amr_parser.parse import AMRParser  # pylint: disable=import-error


@dataclass
class AMROutput:
    """Class to hold AMR Output."""

    label_id: int
    graph: Any
    alignments: Any
    annotations: Any


class AMRModel(object):
    """IBM's Transition AMR Parser (Action Pointer)."""

    def __init__(self, parser: AMRParser) -> None:
        """Initialize AMRModel."""
        self.parser = parser

    @classmethod
    def from_folder(cls, folder: Path) -> "AMRModel":
        """Return an AMRModel object using an AMRParser created from the model data \
        saved in your copy of transition-amr-parser.

        For some reason, the program isn't able to detect the model data \
        if the working directory is not the amr-parser root, even if you provide \
        an absolute path, hence why we change working dirs in this method.
        """
        cdse_path = getcwd()
        # We assume that the checkpoint is in this location within the repo
        in_checkpoint = (
            f"{folder}/DATA/AMR2.0/models"
            "/exp_cofill_o8.3_act-states_RoBERTa-large-top24"
            "/_act-pos-grh_vmask1_shiftpos1_ptr-lay6-h1_grh-lay123-h2-allprev"
            "_1in1out_cam-layall-h2-abuf/ep120-seed42/checkpoint_best.pt"
        )
        chdir(folder)
        parser = AMRParser.from_checkpoint(in_checkpoint)
        chdir(cdse_path)
        return cls(parser)

    def amr_parse_sentences(
        self, sentences: List[List[str]], output_alignments: bool = False
    ) -> AMROutput:
        """Parse sentences in AMR graph and alignments."""
        logging.info(output_alignments)
        annotations = self.parser.parse_sentences(sentences)
        metadata, graph_metadata = Metadata_Parser().readlines(annotations[0][0])
        amr, alignments = AMR_Reader._parse_amr_from_metadata(metadata["tok"], graph_metadata)
        return AMROutput(int(uuid.uuid1()), amr, alignments, annotations)
