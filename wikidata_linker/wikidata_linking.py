"""Collection of Wikidata linking utils."""
import argparse
import json
from pathlib import Path
from typing import Any, Callable, List, Mapping, MutableMapping, Sequence, Set, Tuple, Union

from nltk.stem.snowball import SnowballStemmer
import numpy as np
from pyinflect import getInflection
import requests
from sentence_transformers import SentenceTransformer, util as ss_util
import spacy
from spacy.language import Language
import torch

from wikidata_linker.linker import WikidataLinkingClassifier

CPU = "cpu"
CUDA = "cuda"
nlp = spacy.load("en_core_web_md")


BASE = Path(__file__).parent
MAX_BATCH_SIZE = 8  # I use 8 using GPU, to avoid OOM events. Could probably handle more. Not sure there will be a huge boost if you use CPU.
SOFTMAX = torch.nn.Softmax(dim=1)


STEMMER = SnowballStemmer("english")
OVERLAY_PATH = BASE / "resources/xpo_v3.2_freeze.json"
STEM_MAP_PATH = BASE / "stem_mapping.json"
KGTK_EVENT_CACHE = BASE / "kgtk_event_cache"
KGTK_REFVAR_CACHE = BASE / "kgtk_refvar_cache"
ARG_QNODES_PATH = BASE / "argument_qnodes.json"
EVENT_QNODES_PATH = BASE / "event_qnodes.json"
VERBS_ALL = BASE / "verbs-all.json"
with open(STEM_MAP_PATH, encoding="utf-8") as f:
    STEM_MAP = json.load(f)
with open(OVERLAY_PATH, encoding="utf-8") as json_ontology_fp:
    XPO_ONTOLOGY = json.load(json_ontology_fp)
with open(ARG_QNODES_PATH, encoding="utf-8") as handle:
    ARGUMENT_QNODES = json.load(handle)
with open(EVENT_QNODES_PATH, encoding="utf-8") as handle:
    EVENT_QNODES = json.load(handle)

VERB = "verb"
REFVAR = "refvar"


def get_linker_scores(
    event_description: str,
    use_title: bool,
    candidates: List[MutableMapping[str, Any]],
    linking_model: WikidataLinkingClassifier,
    device: str = CPU,
) -> Any:
    """Gets predictions from Wikidata linking classification model, given a string and candidate JSONs.

    Returns:
        A JSON response.
    """
    i = 0
    scores = []
    candidate_descriptions = []
    for candidate in candidates:
        description = candidate["description"][0] if candidate["description"] else ""
        label = candidate["label"][0] if candidate["label"] else ""
        if use_title:
            description = f"{label} - {description}"
        candidate_descriptions.append(description)
    while i * MAX_BATCH_SIZE < len(candidates):
        candidate_batch = candidate_descriptions[i * MAX_BATCH_SIZE : (i + 1) * MAX_BATCH_SIZE]
        with torch.no_grad():
            logits = (
                linking_model.infer(event_description, candidate_batch)[0].detach()
                if device == CUDA
                else linking_model.infer(event_description, candidate_batch)[0].detach().cpu()
            )
            candidate_batch_scores = SOFTMAX(logits)[:, 2]  # get "entailment" score from model
            for candidate_batch_score in candidate_batch_scores:
                scores.append(candidate_batch_score.item())
            i += 1
    return {"scores": scores}


def get_all_verbs() -> Set[str]:
    """Returns a large set of verbs and each's form.

    Returns:
        A set of strings representing verbs.
    """
    with open(VERBS_ALL, encoding="unicode_escape") as verbs_all_fp:
        all_verbs = json.load(verbs_all_fp)
    flat_list = []
    for verb_forms in all_verbs:
        for verb_form in verb_forms:
            flat_list.append(verb_form)
    return set(flat_list)


verb_set = get_all_verbs()


def get_verb_lemmas(spacy_model: Language, sentence: str) -> Tuple[List[str], List[float]]:
    """Returns the lemma of all verbs (or roots if not verbs found) in the sentence.

    Args:
        spacy_model: A spaCy English Language model.
        sentence: An event description.
        verb_set: A set of verbs and their forms to check words against for additional queries.

    Returns:
        The lemma of the all verbs and words than can occur as verbs, or an empty array if none can be found.
        An additional weight score for each word is returned, discounting words that did not occur as a verb.
    """
    verb_lemmas = []
    extra_lemmas = []
    for token in spacy_model(sentence):
        if token.pos_ == "VERB":
            verb_lemmas.append(token.lemma_)
        elif str(token).lower() in verb_set or str(token.lemma_).lower() in verb_set:
            extra_lemmas.append(token.lemma_)
    return verb_lemmas + extra_lemmas, [1.0] * len(verb_lemmas) + [0.9] * len(extra_lemmas)


def make_cache_path(directory: Path, filename: str) -> Path:
    """Replaces spaces in filename and returns full Path to cache file.

    Args:
        directory: Path to cache directory.
        filename: String for cache filename.

    Returns:
       A Path object pointing to cache file.
    """
    return directory / f"{filename.replace(' ', '_')}.json"


def expand_query_nltk_stem(query: str) -> Sequence[str]:
    """Returns words that share same (or apprixmately same) stem as query.

    Args:
        query: String of KGTK query.

    Returns:
        A list of words that share the same stem as query.
    """
    stemmed_query = STEMMER.stem(query)
    stem_set = []
    if stemmed_query in STEM_MAP:
        stem_set = STEM_MAP[stemmed_query]
    elif len(stemmed_query) > 4:
        if stemmed_query[:-1] in STEM_MAP:
            stem_set = STEM_MAP[stemmed_query[:-1]]
        elif stemmed_query[:-2] in STEM_MAP:
            stem_set = STEM_MAP[stemmed_query[:-2]]
    return stem_set


def make_kgtk_candidates_filter(source_str: str) -> Callable[[Mapping[str, Any]], bool]:
    """Generates filter function for filtering out unwanted candidates for particular string.

    Args:
        source_str: The source_str used to query and to use for filtering candidates.

    Returns:
        A filter function particular to source_str.
    """

    def kgtk_candidate_filter(candidate: Mapping[str, Any]) -> bool:
        str_found = source_str in candidate["label"][0]
        str_found = str_found or any(source_str in alias for alias in candidate["alias"])
        description_and_casematch = candidate["description"] and candidate["label"][0].islower()
        is_qnode = candidate["qnode"].startswith("Q")
        return str_found and description_and_casematch and is_qnode

    return kgtk_candidate_filter


def get_ss_model_similarity(
    ss_model: SentenceTransformer, source_str: str, candidates: List[Mapping[str, Any]]
) -> Any:
    """Computes cosine similarity between source string (event description or refvar) and candidate descriptions.

    Args:
        ss_model: A SentenceTransformer model.
        source_str: The source string for comparing embedding similarity.
        candidates: A list of JSON (dict) objects representing candidates.

    Returns:
        A tensor of similarity scores.
    """
    all_strings = [source_str] + [
        f"{candidate['label'][0]}: {candidate['description'][0]}"
        if candidate["description"]
        else candidate["label"][0]
        for candidate in candidates
    ]
    all_encodings = ss_model.encode(all_strings, convert_to_tensor=True)
    source_emb, candidate_emb = all_encodings[0], all_encodings[1:]
    sim_scores = ss_util.pytorch_cos_sim(source_emb, candidate_emb)[0]
    return sim_scores


def get_request_kgtk(
    query: str,
    cache_file: Path,
    query_item: str = "qnode",
    filter_results: bool = True,
    num_results: int = 20,
) -> List[MutableMapping[str, Any]]:
    """Submits a GET request to KGTK's API.

    Args:
        query: A string to query against KGTK's Wikidata.
        cache_file: The file path to check for cache or write to for caching.
        query_item: A string, either "qnode" or "property", specifying search item.
        filter_results: A boolean indicating whether or not to filter KGTK results.
        num_results: An integer representing how many items to return.

    Returns:
        A list of candidates, or an empty list.
    """
    if "/" in query:
        return []
    if cache_file.is_file():
        with open(cache_file, encoding="utf-8") as cf:
            candidates: List[Any] = json.load(cf)
            return candidates
    kgtk_request_url = "https://kgtk.isi.edu/api"
    params = {
        "q": query,
        "extra_info": "true",
        "language": "en",
        "type": "exact",
        "item": query_item,
        "size": str(num_results),
    }
    try:
        kgtk_response = requests.get(kgtk_request_url, timeout=5, params=params)
    except requests.exceptions.RequestException:
        return []
    if kgtk_response.status_code != 200:
        return []
    else:
        candidates = list(kgtk_response.json())
        for candidate in candidates:
            if not candidate["label"]:
                candidate["label"] = [""]
            if not candidate["description"]:
                candidate["desription"] = [""]
            candidate["kgtk_query"] = query
        if filter_results:
            candidates = list(filter(make_kgtk_candidates_filter(query), candidates))
        with open(cache_file, "w", encoding="utf-8") as cf:
            json.dump(candidates, cf, ensure_ascii=False, indent=2)
            cf.write("\n")
        return candidates


def wikidata_topk(
    ss_model: Union[SentenceTransformer, None],
    source_str: str,
    candidates: List[Mapping[str, Any]],
    k: int,
    thresh: float = 0.15,
) -> Sequence[Mapping[str, Any]]:
    """Returns top k candidates for string according to scoring function.

    Args:
        ss_model: A SentenceTransformer model, or None.
        source_str: A string that candidates will be generated for.
        candidates: A list of JSON (dict) objects representing candidates.
        k: The maximum number of top candidates to return.
        thresh: The minimum cosine similarity to be selected.

    Returns:
        A list of k candidates.
    """
    if ss_model is None:
        scores = np.array([candidate["score"] for candidate in candidates])
    else:
        scores = get_ss_model_similarity(ss_model, source_str, candidates)
    top_k_idx = np.argsort(-scores)[:k]
    top_k_scores = list(zip(top_k_idx, scores[top_k_idx]))
    top_k = [candidates[idx] for idx, score in top_k_scores if score >= thresh]
    return top_k


def filter_duplicate_candidates(
    candidates: List[MutableMapping[str, Any]]
) -> List[MutableMapping[str, Any]]:
    """Filters duplicate candidates from concatenated list of KGTK queries.

    Args:
        candidates: A list of JSON (dict) objects representing candidates.

    Returns:
        A list of unique list of candidates.
    """
    label_description_set = set()
    unique_candidates = []
    for candidate in candidates:
        label_description_tuple = (candidate["label"][0], candidate["description"][0])
        if label_description_tuple not in label_description_set:
            unique_candidates.append(candidate)
            label_description_set.add(label_description_tuple)
    return unique_candidates


def filter_candidates_with_scores(
    scores: List[float], candidates: List[MutableMapping[str, Any]], k: int, thresh: float = 0.01
) -> List[MutableMapping[str, Any]]:
    """Returns top k candidates according to scores and threshold.

    Args:
        scores: A list of floats representing scores for each candidate.
        candidates: A list of JSON (dict) objects representing candidates.
        k: The maximum number of top candidates to return.
        thresh: The minimum cosine similarity to be selected.

    Returns:
        A list of k candidates.
    """
    for (candidate, score) in zip(candidates, scores):
        candidate["linking_score"] = score
    np_scores = np.array(scores)
    top_k_idx = np.argsort(-np_scores)[:k]
    top_k_scores = list(zip(top_k_idx, np_scores[top_k_idx]))
    top_k = [candidates[idx] for idx, score in top_k_scores if score >= thresh]
    return top_k


def disambiguate_verb_kgtk(
    event_description: str,
    linking_model: WikidataLinkingClassifier,
    no_expansion: bool = False,
    k: int = 3,
    device: str = CPU,
) -> Any:
    """Disambiguates verbs from event description and return candidate qnodes.

    Returns:
        A JSON response.
    """
    cleaned_description = event_description.replace("/", " ").replace("_", " ")
    event_verbs, weights = get_verb_lemmas(nlp, cleaned_description)
    kgtk_json: List[MutableMapping[str, Any]] = []
    expanded_weights = []
    for event_verb, weight in zip(event_verbs, weights):
        kgtk_json_len = len(kgtk_json)
        cache_file = make_cache_path(KGTK_EVENT_CACHE, event_verb)
        kgtk_json += get_request_kgtk(event_verb, cache_file)
        event_verb_participle = getInflection(event_verb, tag="VBG")
        if event_verb_participle:
            verb_participle = event_verb_participle[0]
            cache_file = make_cache_path(KGTK_EVENT_CACHE, verb_participle)
            kgtk_json += get_request_kgtk(verb_participle, cache_file)
        if not no_expansion:
            expanded_query = expand_query_nltk_stem(event_verb)
            for query in expanded_query:
                cache_file = make_cache_path(KGTK_EVENT_CACHE, query)
                kgtk_json += get_request_kgtk(query, cache_file)
        expanded_weights += [weight] * (len(kgtk_json) - kgtk_json_len)
    for i, weight in enumerate(expanded_weights):
        kgtk_json[i]["score_weight"] = weight
    unique_candidates = filter_duplicate_candidates(kgtk_json)
    options = []

    candidate_scores = get_linker_scores(
        cleaned_description, False, unique_candidates, linking_model, device
    )["scores"]
    top3 = filter_candidates_with_scores(candidate_scores, unique_candidates, k=k)
    for candidate in top3:
        option = {
            "qnode": candidate["qnode"],
            "rawName": candidate["label"][0],
            "definition": candidate["description"][0],
            "kgtk_query": candidate["kgtk_query"],
        }
        if option not in options:
            options.append(option)
    option_qnodes = [option["qnode"] for option in options]

    other_options = []
    if device == CUDA:
        # Take advantage of the GPU and check the event qnode list
        other_candidate_scores = get_linker_scores(
            cleaned_description, False, EVENT_QNODES, linking_model, device
        )["scores"]
        top3_other_qnodes = filter_candidates_with_scores(other_candidate_scores, EVENT_QNODES, k=k)
        for candidate in top3_other_qnodes:
            # description can be empty sometimes on less popular qnodes
            definition = candidate["description"][0] if candidate["description"] else ""
            option = {
                "qnode": candidate["qnode"],
                "rawName": candidate["label"][0],
                "definition": definition,
            }
            if option not in other_options and option["qnode"] not in option_qnodes:
                other_options.append(option)
    response = {
        "query": event_description,
        "options": options,
        "other_options": other_options,
        "all_options": kgtk_json,
    }
    return response


def disambiguate_refvar_kgtk(
    refvar: str,
    linking_model: WikidataLinkingClassifier,
    context: str = "",
    no_expansion: bool = False,
    k: int = 3,
    device: str = CPU,
) -> Any:
    """Disambiguates refvar with KGTK webserver API.

    Returns:
        A JSON response.
    """
    cleaned_refvar = refvar.replace("/", " ").replace("_", " ")
    cache_file = make_cache_path(KGTK_REFVAR_CACHE, cleaned_refvar)
    kgtk_json = get_request_kgtk(cleaned_refvar, cache_file)
    if len(cleaned_refvar.split()) < 2:
        lemma_refvar = nlp(cleaned_refvar)[0].lemma_
        if lemma_refvar != cleaned_refvar:
            cache_file = make_cache_path(KGTK_REFVAR_CACHE, lemma_refvar)
            kgtk_json += get_request_kgtk(lemma_refvar, cache_file)
        if not no_expansion:
            expanded_query = expand_query_nltk_stem(cleaned_refvar)
            for query in expanded_query:
                cache_file = make_cache_path(KGTK_REFVAR_CACHE, query)
                kgtk_json += get_request_kgtk(query, cache_file)
    if not kgtk_json:
        for token in nlp(cleaned_refvar):
            lemma_token = token.lemma_
            cache_file = make_cache_path(KGTK_REFVAR_CACHE, lemma_token)
            kgtk_json += get_request_kgtk(lemma_token, cache_file)
    unique_candidates = filter_duplicate_candidates(kgtk_json)
    options = []

    if context:
        cleaned_refvar = (
            refvar + " - " + context
        )  # this is how it would be done in MASC, since refvars are not
        # necessarily mentioned in context, otherwise context is probably sufficient
    candidate_scores = get_linker_scores(
        cleaned_refvar, False, unique_candidates, linking_model, device
    )["scores"]
    top3 = filter_candidates_with_scores(candidate_scores, unique_candidates, k=k)
    for candidate in top3:
        # description can be empty sometimes on less popular qnodes
        definition = candidate["description"][0] if candidate["description"] else ""
        option = {
            "qnode": candidate["qnode"],
            "rawName": candidate["label"][0],
            "definition": definition,
        }
        if option not in options:
            options.append(option)
    other_options = []
    if device == CUDA:
        # Take advantage of the GPU and check the argument qnode list
        other_candidate_scores = get_linker_scores(
            cleaned_refvar, False, ARGUMENT_QNODES, linking_model, device
        )["scores"]
        top3_other_qnodes = filter_candidates_with_scores(
            other_candidate_scores, ARGUMENT_QNODES, k=k
        )
        for candidate in top3_other_qnodes:
            # description can be empty sometimes on less popular qnodes
            definition = candidate["description"][0] if candidate["description"] else ""
            option = {
                "qnode": candidate["qnode"],
                "rawName": candidate["label"][0],
                "definition": definition,
            }
            if option not in other_options and option not in options:
                other_options.append(option)
    response = {
        "query": refvar,
        "options": options,
        "other_options": other_options,
        "all_options": kgtk_json,
    }
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query_type",
        type=str,
        help="There type of query that will be launched. Should be 'verb' or 'refvar'",
    )
    parser.add_argument("--query", type=str, help="Term to be queried for")
    parser.add_argument(
        "--context",
        type=str,
        help="Surrounding context for query (context isn't used if no_ss_model specified",
    )
    parser.add_argument(
        "--no_expansion",
        action="store_true",
        help="If specified, no query expansion will be performed (expansion is querying all terms that share stem with query term)",
    )
    parser.add_argument("--k", type=int, default=3, help="Number of options to be returned")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    args = parser.parse_args()
    if args.query_type == VERB:
        print(
            json.dumps(
                disambiguate_verb_kgtk(args.query, args.no_expansion, args.k, args.device)[
                    "options"
                ],
                indent=4,
            )
        )
    elif args.query_type == REFVAR:
        print(
            json.dumps(
                disambiguate_refvar_kgtk(
                    args.query, args.context, args.no_expansion, args.k, args.device
                )["options"],
                indent=4,
            )
        )
    else:
        print("--query_type should be 'verb' or 'refvar'")
