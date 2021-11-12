from abc import abstractmethod
import argparse
from pathlib import Path
from typing import List, Optional

from cdse_covid.claim_detection.models import Claim
from cdse_covid.claim_detection.run_claim_detection import ClaimDataset
from cdse_covid.semantic_extraction.models.amr import AMRFinder
from cdse_covid.semantic_extraction.utils.amr_extraction_utils import identify_x_variable
from cdse_covid.semantic_extraction.models.srl import SRLModel
from cdse_covid.semantic_extraction.entities import XVariable
import spacy

SRL = "srl"
AMR = "amr"


class XVariableFinder(object):
    """ABC for finding the X Variable of a claim."""
    def __init__(self) -> None:
        pass

    @abstractmethod
    def find_x_variable(self, claim: Claim) -> Optional[XVariable]:
        pass


class AMRXVariableFinder(AMRFinder, XVariableFinder):
    """Finds X Variable from the AMR graph and alignments."""
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    def tokenize_sentence(self, text, spacy_tokenizer) -> List[str]:
        tokens = spacy_tokenizer(text.strip())
        return [token.text for token in tokens]
    
    def find_x_variable(self, claim: Claim) -> Optional[XVariable]:
        if not self.amr_graph or not self.amr_alignments:
            sentence = self.tokenize_sentence(claim.claim_sentence, spacy.load("en_core_web_sm").tokenizer)
            breakpoint()
            self.amr_graph, self.amr_alignments = self.model.amr_parse_sentences([sentence])
        return identify_x_variable(self.amr_graph, self.amr_alignments, claim.claim_template)
    

class SRLXVariableFinder(XVariableFinder):
    """Finds X Variable from the SRL graph and alignments."""
    def __init__(self, srl_model: SRLModel) -> None:
        self.srl_model = srl_model
    
    def _reformat_x_variable_in_claim_template(self, claim_template, reference_word="this"):
        """Replaces 'X' in claim template with reference word.
        TODO: Investigate how SRL deals with Person-X, Animal-X, etc."""
        template = []
        for token in claim_template.split():
            if token == "X":
                template.append(reference_word)
            else:
                template.append(token)
        claim_template = " ".join(template)
        return claim_template
        
    def find_x_variable(self, claim: Claim) -> Optional[XVariable]:
        srl_out = self.srl_model.predict(claim.claim_text)
        claim_template = self._reformat_x_variable_in_claim_template(claim.claim_template)
        srl_claim_template = self.srl_model.predict(claim_template)
        arg_label_for_x_variable = [k for k, v in srl_claim_template.args.items() if v == "this"]
        if arg_label_for_x_variable:
            label = arg_label_for_x_variable[0]  # Should only be one
            return srl_out.args.get(label)


def main(input, output, model_type, amr_model, amr_documents):
    claims = ClaimDataset.load_from_dir(input)

    if model_type == SRL:
        model = SRLModel.from_hub("structured-prediction-srl", None)
        finder = SRLXVariableFinder(model)
    elif model_type == AMR:
        if amr_documents:
            finder = AMRXVariableFinder.from_amr_file(amr_documents)
        else:
            finder = AMRXVariableFinder.from_amr_model(amr_model)
    else:
        raise RuntimeError(f"Unknown model type: {model_type}")
    
    for claim in claims:
        x_variable = finder.find_x_variable(claim)
        if x_variable and not claim.x_variable: # If x_variable is not set already, set it
            claim.x_variable = x_variable

    claims.save_to_dir(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-type", choices=[SRL, AMR], default=AMR, type=str)
    parser.add_argument("--amr-model", type=Path, default=None)
    parser.add_argument("--amr-documents", type=Path, default=None)
    parser.add_argument("--domain", type=str, default="covid")
    args = parser.parse_args()

    main(
        args.input,
        args.output,
        args.model_type,
        args.amr_model, 
        args.amr_documents
    )
