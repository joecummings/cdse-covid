"""Run E2E Pegasus claim pipeline."""
import logging
from pathlib import Path
from typing import Optional

from pegasus_wrapper import (
    directory_for,
    initialize_vista_pegasus_wrapper,
    run_python_on_args,
    write_workflow_description,
)
from pegasus_wrapper.artifact import ValueArtifact
from pegasus_wrapper.key_value import ZipKeyValueStore, transform_key_value_store
from pegasus_wrapper.locator import Locator
from pegasus_wrapper.resource_request import SlurmResourceRequest
from saga_tools.conda import CondaConfiguration  # pylint: disable=import-error
import spacy
from vistautils.memory_amount import MemoryAmount
from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

from wikidata_linker.wikidata_linking import CUDA

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def load_and_serialize_spacy_model(
    path_to_saved_model: Path, model: str = "en_core_web_md"
) -> None:
    """Load a saved SpaCy model and serialized it."""
    spacy_model = spacy.load(model)
    spacy_model.to_disk(path_to_saved_model)


def main(params: Parameters) -> None:
    """Entrypoint to claims pipeline."""
    initialize_vista_pegasus_wrapper(params)

    # Base info
    base_locator = Locator(("claims",))
    input_corpus_dir = params.existing_directory("corpus")
    from_raw_documents = params.boolean("from_raw_documents", default=True)
    state_dict = params.existing_file("state_dict")

    spacy_params = params.namespace("spacy")
    model_path = directory_for(base_locator) / "spacy.mdl"
    load_and_serialize_spacy_model(model_path, spacy_params.string("model_type"))

    if from_raw_documents:
        preprocessed_docs = annotate_raw_documents(
            spacy_params, base_locator, model_path, input_corpus_dir
        )
        claim_detection_output = isi_claim_detection(
            params, base_locator, model_path, preprocessed_docs
        )
    else:
        uiuc_claims = ingest_uiuc_claims(params, base_locator)
        claim_detection_output = add_topic_information(params, base_locator, uiuc_claims)

    # Ingest EDL data
    edl_params = params.namespace("edl")
    edl_locator = base_locator / "edl"
    edl_output = ingest_edl_data(edl_params, edl_locator)

    # AMR parsing over the entirety of each document
    amr_params = params.namespace("amr")
    amr_max_tokens = amr_params.integer("max_tokens", default=50)
    device = amr_params.string("device")
    if params.string("site") == "saga":
        larger_resource = SlurmResourceRequest(memory=MemoryAmount.parse("8G"), num_gpus=1)
        even_larger_resource = SlurmResourceRequest(memory=MemoryAmount.parse("16G"), num_gpus=1)
        device = CUDA
    else:
        larger_resource = None
        even_larger_resource = None

    if amr_params.optional_existing_file("python_file_all"):
        _ = amr_over_all_docs(
            params,
            input_corpus_dir,
            base_locator,
            amr_params,
            amr_max_tokens,
            larger_resource,
        )

    # AMR parsing for claims
    amr_loc = base_locator / "amr"

    def amr_as_func(kvs: ZipKeyValueStore) -> ZipKeyValueStore:
        amr_python_file = amr_params.existing_file("python_file")
        output_locator = kvs.locator / "amr"
        amr_output_dir = directory_for(output_locator) / "documents.zip"
        amr_job = run_python_on_args(
            output_locator,
            amr_python_file,
            f"""
            --input {kvs.path} \
            --output {amr_output_dir} \
            --amr-parser-model {amr_params.existing_directory("model_path")} \
            --max-tokens {amr_max_tokens} \
            --state-dict {state_dict} \
            --domain {amr_params.string("domain", default="general")} \
            --device {device}
            """,
            override_conda_config=CondaConfiguration(
                conda_base_path=params.existing_directory("conda_base_path"),
                conda_environment="transition-amr-parser",
            ),
            resource_request=even_larger_resource,
            depends_on=[kvs],
        )
        return ZipKeyValueStore(path=amr_output_dir, depends_on=[amr_job], locator=output_locator)

    amr_output = transform_key_value_store(
        claim_detection_output,
        amr_as_func,
        output_locator=amr_loc,
        parallelism=10,
    )

    # SRL
    srl_loc = base_locator / "srl"

    def run_srl(kvs: ZipKeyValueStore) -> ZipKeyValueStore:
        srl_params = params.namespace("srl")
        srl_python_file = srl_params.existing_file("python_file")
        srl_output_dir = directory_for(srl_loc) / "documents.zip"
        srl_job = run_python_on_args(
            kvs.locator,
            srl_python_file,
            f"""
            --input {kvs.path} \
            --output {srl_output_dir} \
            --spacy-model {model_path} \
            """,
            depends_on=[kvs],
        )
        return ZipKeyValueStore(path=srl_output_dir, depends_on=[srl_job], locator=srl_loc)

    srl_output = transform_key_value_store(
        amr_output, run_srl, output_locator=srl_loc, parallelism=10
    )

    wikidata_loc = base_locator / "wikidata"

    def run_wikidata(kvs: ZipKeyValueStore) -> ZipKeyValueStore:
        wikidata_params = params.namespace("wikidata")
        wikidata_python_file = wikidata_params.existing_file("python_file")
        wikidata_output_dir = directory_for(wikidata_loc) / "documents.zip"
        wikidata_job = run_python_on_args(
            kvs.locator,
            wikidata_python_file,
            f"""
            --claim-input {kvs.path} \
            --state-dict {state_dict} \
            --output {wikidata_output_dir} \
            --device {device}
            """,
            override_conda_config=CondaConfiguration(
                conda_base_path=params.existing_directory("conda_base_path"),
                conda_environment="transition-amr-parser",
            ),
            resource_request=larger_resource,
            depends_on=[kvs],
        )
        return ZipKeyValueStore(path=wikidata_output_dir, depends_on=[wikidata_job])

    wikidata_output = transform_key_value_store(
        srl_output, run_wikidata, output_locator=wikidata_loc, parallelism=10
    )

    # # Entity unification
    entity_loc = edl_locator / "edl_unified"

    def unify_entities(kvs: ZipKeyValueStore, *, output_locator: Locator = None) -> ZipKeyValueStore:
        qnode_freebase_file = edl_params.existing_file("qnode_freebase_file")
        freebase_to_qnodes = edl_params.creatable_file("freebase_to_qnodes")
        ent_python_file = edl_params.existing_file("ent_unification")
        ent_output_dir = directory_for(entity_loc) / "documents.zip"
        include_contains = edl_params.boolean("include_contains")
        ent_unify_job = run_python_on_args(
            kvs.locator,
            ent_python_file,
            f"""
            --edl {edl_output.value} \
            --qnode-freebase {qnode_freebase_file} \
            --freebase-to-qnodes {freebase_to_qnodes} \
            --claims {kvs.path} \
            --output {ent_output_dir} \
            {'--include-contains' if include_contains else ''}
            """,
            depends_on=[kvs],
        )
        return ZipKeyValueStore(path=ent_output_dir, depends_on=[ent_unify_job])

    entities_output = transform_key_value_store(
        wikidata_output, unify_entities, output_locator=entity_loc, parallelism=1
    )

    # Unify everything
    unify_params = params.namespace("unify")
    unify_loc = base_locator / "unify"
    unify_python_job = unify_params.existing_file("python_file")
    output_file = unify_params.creatable_file("output")
    run_python_on_args(
        unify_loc,
        unify_python_job,
        f"""
        --input {entities_output.path} \
        --output {output_file} \
        """,
        depends_on=[entities_output],
    )

    write_workflow_description()


def ingest_edl_data(edl_params: Parameters, edl_locator: Locator) -> ValueArtifact:
    """Ingest EDL data from UIUC and return mapping as ValueArtifact."""
    edl_ingester = edl_params.existing_file("ingester")
    edl_mapping_file = directory_for(edl_locator) / "edl_mapping.pkl"
    edl_final = edl_params.existing_directory("edl_output_dir")
    edl_job = run_python_on_args(
        edl_locator,
        edl_ingester,
        f"""
        --edl-output {edl_final} \
        --output {edl_mapping_file}
        """,
        depends_on=[],
    )
    return ValueArtifact(value=edl_mapping_file, depends_on=[edl_job])


def add_topic_information(
    params: Parameters, base_locator: Locator, uiuc_claims: ValueArtifact
) -> ZipKeyValueStore:
    """Add topic information to Claims."""
    topic_params = params.namespace("topic_information")
    topic_locator = base_locator / "topic_info"
    add_topic_python_file = topic_params.existing_file("python_file")
    ss_model = topic_params.string("ss_model")
    topics = topic_params.existing_file("patterns")
    out = directory_for(topic_locator) / "documents.zip"
    add_topics_job = run_python_on_args(
        topic_locator,
        add_topic_python_file,
        f"""
            --claims {uiuc_claims.path} \
            --templates-file {topics} \
            --ss-model {ss_model} \
            --output {out}
            """,
        depends_on=[uiuc_claims],
    )
    return ZipKeyValueStore(path=out, depends_on=[add_topics_job], locator=base_locator / "corpus")


def ingest_uiuc_claims(params: Parameters, base_locator: Locator) -> ZipKeyValueStore:
    """Ingest UIUC claims."""
    uiuc_params = params.namespace("uiuc_claims")
    uiuc_locator = base_locator / "uiuc_claims"
    uiuc_ingester = uiuc_params.existing_file("ingester")
    uiuc_input = uiuc_params.existing_file("json_file_of_claims")
    uiuc_output_dir = directory_for(uiuc_locator) / "documents.zip"
    uiuc_job = run_python_on_args(
        uiuc_locator,
        uiuc_ingester,
        f"""
            --claims-file {uiuc_input} \
            --output {uiuc_output_dir}
            """,
        depends_on=[],
    )
    return ZipKeyValueStore(
        path=uiuc_output_dir, depends_on=[uiuc_job], locator=base_locator / "corpus"
    )


def annotate_raw_documents(
    params: Parameters, base_locator: Locator, model_path: Path, input_corpus_dir: Path
) -> ValueArtifact:
    """Annotate raw documents with SpaCy tags."""
    spacy_python_file = params.existing_file("ingester")
    spacified_docs_locator = base_locator / "spacified"
    spacified_output_dir = directory_for(spacified_docs_locator)
    preprocess_job = run_python_on_args(
        spacified_docs_locator,
        spacy_python_file,
        f"--corpus {input_corpus_dir} --output {spacified_output_dir} --spacy-model {model_path}",
        depends_on=[],
    )
    return ValueArtifact(value=spacified_output_dir, depends_on=[preprocess_job])


def amr_over_all_docs(
    params: Parameters,
    input_corpus_dir: Path,
    base_locator: Locator,
    amr_params: Parameters,
    amr_max_tokens: int,
    larger_resource: Optional[SlurmResourceRequest],
) -> ValueArtifact:
    """Run AMR parser over all sentences in all docs."""
    amr_all_loc = base_locator / "amr_all"
    amr_all_python_file = amr_params.existing_file("python_file_all")
    amr_all_output_dir = directory_for(amr_all_loc) / "documents"
    amr_all_job = run_python_on_args(
        amr_all_loc,
        amr_all_python_file,
        f"""
        --corpus {input_corpus_dir} \
        --output {amr_all_output_dir} \
        --amr-parser-model {amr_params.existing_directory("model_path")} \
        --max-tokens {amr_max_tokens}
        """,
        override_conda_config=CondaConfiguration(
            conda_base_path=params.existing_directory("conda_base_path"),
            conda_environment="transition-amr-parser",
        ),
        resource_request=larger_resource,
        depends_on=[],
    )
    return ValueArtifact(value=amr_all_output_dir, depends_on=[amr_all_job])


def isi_claim_detection(
    params: Parameters, base_locator: Locator, model_path: Path, preprocessed_docs: ValueArtifact
) -> ValueArtifact:
    """Detect claims from SpaCy docs using ISI claim detection."""
    claim_params = params.namespace("claim_detection")
    claim_loc = base_locator / "claim_detection"
    claim_python_file = claim_params.existing_file("python_file")
    patterns_file = claim_params.existing_file("patterns")
    claim_output_dir = directory_for(claim_loc) / "documents"
    claim_detection_job = run_python_on_args(
        claim_loc,
        claim_python_file,
        f"""
        --input {preprocessed_docs.value} \
        --patterns {patterns_file} \
        --out {claim_output_dir} \
        --spacy-model {model_path}
        """,
        depends_on=[preprocessed_docs],
    )
    return ValueArtifact(value=claim_output_dir, depends_on=[claim_detection_job])


if __name__ == "__main__":
    parameters_only_entry_point(main)
