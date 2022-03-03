"""Classes related to AMR Model."""
import logging
from os import chdir, getcwd
from pathlib import Path
from typing import List, Optional
import uuid

from amr_utils.amr_readers import AMR_Reader, Metadata_Parser
from transition_amr_parser.parse import AMRParser

from cdse_covid.semantic_extraction.models.output_formats import AMROutput


class AMRModel(object):
    """IBM's Transition AMR Parser (Action Pointer)."""

    def __init__(self, parser: AMRParser) -> None:
        """Initialize AMRModel."""
        self.parser = parser

    @classmethod
    def from_folder(cls, folder: Path) -> "AMRModel":
        """Return an AMRModel object using an AMRParser created from the model data saved in your copy of transition-amr-parser.

        For some reason, the program isn't able to detect the model data
        if the working directory is not the amr-parser root, even if you provide
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
    ) -> Optional[AMROutput]:
        """Parse sentences in AMR graph and alignments."""
        logging.info(output_alignments)
        try:
            annotations = self.parser.parse_sentences(sentences)
        except IndexError as e:
            logging.warning(e)
            return None
        metadata, graph_metadata = Metadata_Parser().readlines(annotations[0][0])
        # Make sure there's a graph we can work with
        if not graph_metadata.get("node"):
            return None
        amr, alignments = AMR_Reader._parse_amr_from_metadata(metadata["tok"], graph_metadata)
        return AMROutput(int(uuid.uuid1()), amr, alignments, annotations)
