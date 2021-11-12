from dataclasses import dataclass
import logging
from typing import Any, List, MutableMapping
import uuid

from allennlp_models.pretrained import load_predictor  # pylint: disable=import-error
from spacy.language import Language

from cdse_covid.semantic_extraction.utils.claimer_utils import LEMMATIZER


@dataclass
class SRLOutput:
    label_id: int
    verb: str
    args: MutableMapping[str, str]


class SRLModel:
    def __init__(self, predictor: Any, *, spacy_model: Language) -> None:
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

    def _clean_srl_output(self, srl_output: MutableMapping[str, Any]) -> MutableMapping[str, str]:
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
            tag_sequences: MutableMapping[str, str] = {}
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

    def _stem_verb(self, verbs: List[MutableMapping[str, Any]]) -> str:
        verb_word = verbs[0]["verb"]
        return str(LEMMATIZER.lemmatize(verb_word, pos="v"))

    def predict(self, sentence: str) -> SRLOutput:
        """Predict SRL over a sentence."""
        roles = self.predictor.predict(sentence)
        verb = ""
        if len(roles["verbs"]) > 1:
            logging.warning("More than one main verb in instance: %s", sentence)
        elif len(roles["verbs"]) < 1:
            logging.warning("No verbs detected in sentence: %s", sentence)
        else:
            verb = self._stem_verb(roles["verbs"])
        args = self._clean_srl_output(roles)
        return SRLOutput(int(uuid.uuid1()), verb=verb, args=args)
