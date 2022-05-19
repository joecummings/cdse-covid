"""Take in detected claims and add topic/subtopic/template information."""
import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm

from cdse_covid.claim_detection.run_claim_detection import ClaimDataset


class TemplateIdentifier(object):
    """Class to identify closest template to a given claim."""

    def __init__(self, similarity_model: SentenceTransformer, templates: List[str]) -> None:
        """Initialize TemplateIdentifier class."""
        self.similarity_model = similarity_model
        self.templates = templates
        self._encode_templates(templates)

    def _encode_templates(self, templates: List[str]) -> None:
        """Encode templates using the SentenceBERT model."""
        self.encoded_templates = self.similarity_model.encode(templates, convert_to_tensor=True)

    def identify_template(self, claim_text: str) -> Tuple[str, float]:
        """Encode and compare claim text to encoded templates to find most similar one."""
        encoded_text = self.similarity_model.encode(claim_text, convert_to_tensor=True)
        cos_matrix = util.pytorch_cos_sim(encoded_text, self.encoded_templates)
        sim_row = cos_matrix[0].cpu()
        # Currently taking the highest value but we might want to have a None if all of them suck
        return self.templates[int(np.argmax(sim_row))], float(torch.max(sim_row))


def main(claims: Path, templates_file: Path, ss_model: str, output: Path) -> None:
    """Entypoint for adding topic information to incomplete Claims."""
    model = SentenceTransformer(ss_model)
    topics_info = {}
    with open(templates_file, "r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            key = row[3]
            topics_info[key] = {"topic": row[2], "subtopic": row[1]}
    template_identifier = TemplateIdentifier(model, list(topics_info.keys()))

    claims_dataset = ClaimDataset.load_from_key_value_store(claims)
    for claim in tqdm(claims_dataset, total=len(claims_dataset.claims)):
        template, confidence = template_identifier.identify_template(claim.claim_text)
        claim.claim_template = template
        claim.topic = topics_info[claim.claim_template]["topic"]
        claim.subtopic = topics_info[claim.claim_template]["subtopic"]

        claim.add_theory("template_confidence_score", confidence)

    claims_dataset.save_to_key_value_store(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--claims", type=Path, help="Path to claims")
    p.add_argument("--templates-file", type=Path, help="Path to topics/subtopics/templates")
    p.add_argument("--ss-model", type=str, help="Type of SS model to use")
    p.add_argument("--output", type=Path, help="Output path for updated claims")
    args = p.parse_args()
    main(args.claims, args.templates_file, args.ss_model, args.output)
