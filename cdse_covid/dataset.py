import logging
from pathlib import Path
import spacy
from spacy.language import Language
from spacy.tokens import Doc
from spacy.vocab import Vocab # pylint: disable=no-name-in-module
from typing import Sequence, Tuple
from spacy.attrs import HEAD, SENT_START


class AIDADataset:
    def __init__(self, documents: Sequence[Tuple[str, Doc]]) -> None:
        self.documents = documents
        self.templates = None

    @classmethod
    def from_text_files(
        cls, path_to_text_files: Path, *, nlp: Language
    ) -> "AIDADataset":
        all_docs = []
        for txt_file in path_to_text_files.glob("*.txt"):
            with open(txt_file, "r") as handle:
                doc_text = handle.read()
                parsed_english = nlp(doc_text)
                all_docs.append((txt_file.stem, parsed_english))
        return cls(all_docs)

    @classmethod
    def from_serialized_docs(cls, path_to_docs: Path) -> "AIDADataset":
        all_docs = []
        for doc in path_to_docs.glob("*.spacy"):
            try:
                spacy_doc = Doc(Vocab()).from_disk(doc)
            except ValueError as e:
                logging.warning(e)
                spacy_doc = Doc(Vocab()).from_disk(doc, exclude=[SENT_START])
            all_docs.append((doc.stem, spacy_doc))
        return cls(all_docs)

    def load_templates(self, templates_file: Path):
        templates = []
        with open(templates_file, "r") as handle:
            for i, line in enumerate(handle):
                if i != 0:
                    split_line = line.split("\t")
                    templates.append(split_line[3].replace("\n", ""))
        self.templates = templates
