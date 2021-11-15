"""Collection of Wikidata linking utils."""
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Callable, List, MutableMapping, Sequence, Tuple, Union

from nltk.stem.snowball import SnowballStemmer
import numpy as np
import requests

try:
    from sentence_transformers import SentenceTransformer, util as ss_util
except ModuleNotFoundError:
    logging.warning("Cannot load sentence transformers in this env.")
import torch

STEMMER = SnowballStemmer("english")

BASE = Path(__file__).parent
PRETRAINED_MODEL = BASE / "sent_model"
STEM_MAP_PATH = BASE / "stem_mapping.json"
KGTK_CACHE = BASE / "kgtk_cache"
with open(STEM_MAP_PATH, "r", encoding="utf-8") as f:
    STEM_MAP = json.load(f)


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


def make_kgtk_candidates_filter(source_str: str) -> Callable[[MutableMapping[str, Any]], bool]:
    """Generates filter function for filtering out unwanted candidates for particular string.

    Args:
        source_str: The source_str used to query and to use for filtering candidates.

    Returns:
        A filter function particular to source_str.
    """

    def kgtk_candidate_filter(candidate: MutableMapping[str, Any]) -> bool:
        str_found = source_str in candidate["label"][0]
        str_found = str_found or any(source_str in alias for alias in candidate["alias"])
        description_and_casematch = candidate["description"] and candidate["label"][0].islower()
        return str_found and description_and_casematch

    return kgtk_candidate_filter


def get_ss_model_similarity(
    ss_model: SentenceTransformer, source_str: str, candidates: List[MutableMapping[str, Any]]
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
    return ss_util.pytorch_cos_sim(source_emb, candidate_emb)[0]


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
    if cache_file.is_file():
        with open(cache_file, "r", encoding="utf-8") as file:
            candidates: List[Any] = json.load(file)
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
    candidates = list(kgtk_response.json())
    for candidate in candidates:
        if not candidate["label"]:
            candidate["label"] = [""]
    if filter_results:
        candidates = list(filter(make_kgtk_candidates_filter(query), candidates))
    with open(cache_file, "w", encoding="utf-8") as file:
        json.dump(candidates, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return candidates


def wikidata_topk(
    ss_model: Union[SentenceTransformer, None],
    source_str: str,
    candidates: List[MutableMapping[str, Any]],
    k: int,
    thresh: float = 0.15,
) -> Sequence[MutableMapping[str, Any]]:
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
    # Without this block, wikidata linking crashes if running on a GPU
    if isinstance(scores, torch.Tensor):  # type: ignore
        scores = scores.cpu().numpy()  # type: ignore
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
        if candidate["description"]:
            label_description_tuple: Tuple[Any, ...] = tuple(
                (candidate["label"][0], candidate["description"][0])
            )
            if label_description_tuple not in label_description_set:
                unique_candidates.append(candidate)
                label_description_set.add(label_description_tuple)
    return unique_candidates


def request_top_n_qnodes(
    description: str,
    *,
    n: int,
    ss_model: SentenceTransformer,
    embeddings: torch.FloatTensor,  # pylint: disable=no-member
    qnodes: Sequence[MutableMapping[str, Any]],
) -> Sequence[MutableMapping[str, Any]]:
    """Get the top *n* predicted qnodes from the *ss_model* provided.

    Arguments:
        description: Text to be the basis of the prediction.
        n: Number of top predictions to be returned.
        ss_model: SentenceTransformer model to make the predictions.
        embeddings: Embeddings of a set of qnodes.
        qnodes: Sequence of qnode dictionaries.

    Returns:
        List of predictions (in order of most similar -> least similar) of qnodes.
    """
    # Similarity scoring
    event_embedding = ss_model.encode([description], convert_to_tensor=True)
    cosine_scores = ss_util.pytorch_cos_sim(event_embedding, embeddings)
    sorted_indices = cosine_scores.argsort(descending=True)
    return [qnodes[idx] for idx in sorted_indices[0][:n]]


def disambiguate_kgtk(
    context: str,
    query: str,
    no_ss_model: bool = False,
    no_expansion: bool = False,
    k: int = 3,
    thresh: float = 0.15,
) -> MutableMapping[str, Any]:
    """Disambiguates verbs from event description and return candidate qnodes.

    Returns:
        A JSON response.
    """
    kgtk_json = []
    cache_file = make_cache_path(KGTK_CACHE, query)
    kgtk_json += get_request_kgtk(query, cache_file, filter_results=False)
    if not no_expansion:
        expanded_query = expand_query_nltk_stem(query)
        for q in expanded_query:
            cache_file = make_cache_path(KGTK_CACHE, q)
            kgtk_json += get_request_kgtk(q, cache_file)
    unique_candidates = filter_duplicate_candidates(kgtk_json)
    options = []
    if no_ss_model:
        top3 = wikidata_topk(None, context, unique_candidates, k, thresh=thresh)
    else:
        ss_model = SentenceTransformer(str(PRETRAINED_MODEL))
        top3 = wikidata_topk(ss_model, context, unique_candidates, k, thresh=thresh)
    for candidate in top3:
        option = {
            "qnode": candidate["qnode"],
            "rawName": candidate["label"][0],
            "definition": candidate["description"][0] if candidate["description"] else "",
        }
        if option not in options:
            options.append(option)
    return {
        "context": context,
        "query": query,
        "options": options,
        "all_options": kgtk_json,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no_expansion",
        action="store_true",
        help="If specified, no query expansion will be performed (expansion is querying all terms that share stem with query term)",
    )
    parser.add_argument(
        "--no_ss_model",
        action="store_true",
        help="If specified, will use score from KGTK rather than cosine similarity (withing MASC, this is often useful when dealing with refvars (like common nouns), but not for event verbs)",
    )
    parser.add_argument("--k", type=int, default=3, help="Number of options to be returned")
    parser.add_argument(
        "--thresh",
        type=float,
        default=0.15,
        help="Threshold score to be used to filter out unlikely scores. We use 0.15 in MACS. If using no_ss_model, scale will be different and harder to tune",
    )
    parser.add_argument("--query", type=str, help="Term to be queried for")
    parser.add_argument(
        "--context",
        type=str,
        help="Surrounding context for query (context isn't used if no_ss_model specified",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            disambiguate_kgtk(
                args.context, args.query, args.no_ss_model, args.no_expansion, args.k, args.thresh
            )["options"],
            indent=4,
        )
    )
