from typing import Any, Mapping

from spacy.language import Language


def clean_srl_output(srl_output: Mapping[str, Any], nlp: Language) -> Mapping[str, str]:
    """Takes AllenNLP SRL output and returns list of ARGN strings.

    Args:
        srl_output: A string of semantic role labelling output

    Returns:
        A list of ARG strings from SRL output.
    """
    cleaned_output = []
    for verb in srl_output["verbs"]:
        if "tags" in verb and len(verb["tags"]) > 0:
            tag_sequences: Mapping[str, str] = {}
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
                cleaned_sequence = remove_leading_trailing_stopwords(nlp, sequence)
                if cleaned_sequence:
                    cleaned_output.append(cleaned_sequence)
                    tag_sequences[tag] = cleaned_sequence
    # logging.warning("Not cleaning sequences.")
    return tag_sequences


def remove_leading_trailing_stopwords(nlp: Language, string: str) -> str:
    """Returns string with leading and trailing stopwords removed.

    Args:
        nlp: A spaCy English Language model.
        string: Text to have stopwords removed.

    Returns:
        A string with no leading or trailing stopwords included.
    """
    doc = nlp(string)
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
