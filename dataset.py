from pathlib import Path
from spacy.language import Language
from spacy.tokens import Doc
from typing import Sequence, Dict, List

from amr_utils.amr_readers import AMR_Reader
from amr_utils.amr import AMR

AMR_READER = AMR_Reader()


class AIDADataset:
    def __init__(self, documents: Sequence[Doc], *, templates_file: Path = None) -> None:
        self.documents = documents
        self.templates = None
        self.load_templates(templates_file)

    @classmethod
    def from_text_files(cls, path_to_text_files: Path, *, nlp: Language) -> "AIDADataset":
        all_docs = []
        for txt_file in path_to_text_files.glob("*.txt"):
            with open(txt_file, "r") as handle:
                doc_text = handle.read()
                parsed_english = nlp(doc_text)
                all_docs.append(parsed_english)
        return cls(all_docs)

    def load_templates(self, templates_file: Path):
        templates = []
        with open(templates_file, "r") as handle:
            for i, line in enumerate(handle):
                if i != 0:
                    split_line = line.split("\t")
                    templates.append(split_line[3].replace("\n", ""))
        self.templates = templates

    def _batch_convert_amr_to_spacy(self, amr_files: Path) -> Dict[Doc, List[AMR]]:
        docs_to_amrs = {}
        for _file in amr_files.glob("*.amr"):
            amrs = AMR_READER.load(_file, remove_wiki=True)
            sentences = [
                graph.metadata["snt"]
                for graph in amrs
            ]
            doc_text = " ".join(sentences)
            parsed_english = self.nlp(doc_text)
            docs_to_amrs[parsed_english] = amrs
        return docs_to_amrs
