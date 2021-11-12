import logging
from pathlib import Path
from typing import List, Sequence, Tuple

from spacy.attrs import HEAD, SENT_START  # pylint: disable=no-name-in-module
from spacy.language import Language
from spacy.tokens import Doc  # pylint: disable=no-name-in-module
from spacy.vocab import Vocab  # pylint: disable=no-name-in-module


class AIDADataset:
    def __init__(self, documents: Sequence[Tuple[str, Doc]]) -> None:
        self.documents = documents
        self.templates: List[str] = []

    @classmethod
    def from_text_files(cls, path_to_text_files: Path, *, nlp: Language) -> "AIDADataset":
        all_docs = []
        for txt_file in path_to_text_files.glob("*.txt"):
            with open(txt_file, "r", encoding="utf-8") as handle:
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
                spacy_doc = Doc(Vocab()).from_disk(doc, exclude=[SENT_START, HEAD])
            all_docs.append((doc.stem, spacy_doc))
        return cls(all_docs)

    def load_templates(self, templates_file: Path) -> None:
        templates = []
        with open(templates_file, "r", encoding="utf-8") as handle:
            for i, line in enumerate(handle):
                if i != 0:
                    split_line = line.split("\t")
                    templates.append(split_line[3].replace("\n", ""))
        self.templates = templates
