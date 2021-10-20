import argparse
import logging
from pathlib import Path
from typing import Any, Mapping
import uuid

from allennlp_models.pretrained import load_predictor
from nltk.stem.snowball import SnowballStemmer
from cdse_covid.claim_detection.run_claim_detection import ClaimDataset
import spacy
from spacy.language import Language

from cdse_covid.semantic_extraction.models import SRLabel

from cdse_covid.semantic_extraction.claimer_utils import LEMMATIZER

class SRLModel:
    def __init__(self, predictor, *, spacy_model) -> None:
        self.predictor = predictor
        self.spacy_model = spacy_model

    @classmethod
    def from_hub(cls, filename: str, spacy_model: Language) -> "SRLModel":
        return cls(load_predictor(filename), spacy_model=spacy_model)

    def _remove_leading_trailing_stopwords(self, string: str) -> str:
        """Returns string with leading and trailing stopwords removed.

        Args:
            string: Text to have stopwords removed.

        Returns:
            A string with no leading or trailing stopwords included.
        """
        doc = self.spacy_model(string)
        first_nonstop_idx = -1
        last_nonstop_idx = -1
        for i, token in enumerate(doc):
            if first_nonstop_idx == -1 and not token.is_stop:
                first_nonstop_idx = i
            if first_nonstop_idx > -1 and not token.is_stop:
                last_nonstop_idx = i
        clipped_doc = doc[first_nonstop_idx : last_nonstop_idx + 1]
        # Want to ignore overly long argument phrases (although probably uncommon)
        if len(clipped_doc) > 4:
            return ""
        return str(clipped_doc)

    def _clean_srl_output(self, srl_output: Mapping[str, Any]) -> Mapping[str, str]:
        """Takes AllenNLP SRL output and returns list of ARGN strings.

        Args:
            srl_output: A string of semantic role labelling output

        Returns:
            A list of ARG strings from SRL output.
        """
        if not srl_output["verbs"]:
            return dict()

        cleaned_output = []
        for verb in srl_output["verbs"]:
            tag_sequences: Mapping[str, str] = {}
            if "tags" in verb and len(verb["tags"]) > 0:
                tags_words = zip(
                    [tag.split("-", 1)[-1] for tag in verb["tags"]], srl_output["words"]
                )
                for tag, word in tags_words:
                    if "ARG" in tag:
                        if tag in tag_sequences:
                            tag_sequences[tag] += f" {word}"
                        else:
                            tag_sequences[tag] = word
                for tag, sequence in tag_sequences.items():
                    cleaned_sequence = self._remove_leading_trailing_stopwords(sequence)
                    if cleaned_sequence:
                        cleaned_output.append(cleaned_sequence)
                        tag_sequences[tag] = cleaned_sequence
        return tag_sequences

    def _stem_verb(self, verbs):
        verb_word = verbs[0]["verb"]
        return LEMMATIZER.lemmatize(verb_word, pos="v")

    def predict(self, sentence: str) -> SRLabel:
        """Predict SRL over a sentence."""
        roles = self.predictor.predict(sentence)
        verb = None
        if len(roles["verbs"]) > 1:
            logging.warning("More than one main verb in instance: %s", sentence)
        elif len(roles["verbs"]) < 1:
            logging.warning("No verbs detected in sentence: %s", sentence)
        else:
            verb = self._stem_verb(roles["verbs"])
        args = self._clean_srl_output(roles)
        return SRLabel(int(uuid.uuid1()), verb=verb, args=args)


def reformat_x_variable_in_claim_template(claim_template, reference_word="this"):
    """Replaces 'X' in claim template with reference word.
    
    TODO: Investigate how SRL deals with Person-X, Animal-X, etc.
    """
    template = []
    for token in claim_template.split():
        if token == "X":
            template.append(reference_word)
        else:
            template.append(token)
    claim_template = " ".join(template)
    return claim_template


def main(inputs, output, *, spacy_model):
    srl_model = SRLModel.from_hub("structured-prediction-srl", spacy_model)
    claim_ds = ClaimDataset.load_from_dir(inputs)

    for claim in claim_ds.claims:
        srl_out = srl_model.predict(claim.claim_text)
        
        # Add claim semantics
        claim.claim_semantics = {
            "event": srl_out.verb,
            "args": srl_out.args
        }

        # Find X variable
        claim_template = reformat_x_variable_in_claim_template(claim.claim_template)
        srl_claim_template = srl_model.predict(claim_template)
        arg_label_for_x_variable = [k for k, v in srl_claim_template.args.items() if v == "this"]
        if arg_label_for_x_variable:
            label = arg_label_for_x_variable[0] # Should only be one
            x_variable = srl_out.args.get(label)
            if x_variable:
                claim.x_variable = x_variable

        claim.add_theory("srl", srl_out)
    claim_ds.save_to_dir(output)

    logging.info("Finished saving SRL labels to %s.", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Input docs", type=Path)
    parser.add_argument("--output", help="Out file", type=Path)
    parser.add_argument("--spacy-model", type=Path)
    args = parser.parse_args()

    model = spacy.load(args.spacy_model)

    main(args.input, args.output, spacy_model=model)
