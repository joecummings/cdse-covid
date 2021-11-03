from pathlib import Path
from pegasus_wrapper.artifact import ValueArtifact
from pegasus_wrapper.resource_request import SlurmResourceRequest
from saga_tools.conda import CondaConfiguration
import spacy
import logging

from pegasus_wrapper import (
    directory_for,
    initialize_vista_pegasus_wrapper,
    write_workflow_description,
    run_python_on_args,
)
from pegasus_wrapper.locator import Locator
from vistautils.memory_amount import MemoryAmount
from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def load_and_serialize_spacy_model(
    path_to_saved_model: Path, model="en_core_web_sm"
) -> None:
    spacy_model = spacy.load(model)
    spacy_model.to_disk(path_to_saved_model)


def main(params: Parameters):
    initialize_vista_pegasus_wrapper(params)

    # Base info
    base_locator = Locator(("claims",))
    input_corpus_dir = params.existing_directory("corpus")

    # Preprocessing
    spacy_params = params.namespace("spacy")
    spacy_python_file = spacy_params.existing_file("ingester")
    spacified_docs_locator = base_locator / "spacified"
    model_path = directory_for(base_locator) / "spacy.mdl"
    load_and_serialize_spacy_model(model_path, spacy_params.string("model_type"))
    spacified_output_dir = directory_for(spacified_docs_locator)
    preprocess_job = run_python_on_args(
        spacified_docs_locator,
        spacy_python_file,
        f"--corpus {input_corpus_dir} --output {spacified_output_dir} --spacy-model {model_path}",
        depends_on=[],
    )
    preprocessed_docs = ValueArtifact(
        value=spacified_output_dir, depends_on=[preprocess_job]
    )

    # Claim detection
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
        depends_on=[preprocessed_docs]
    )
    claim_detection_output = ValueArtifact(
        value=claim_output_dir, depends_on=[claim_detection_job]
    )

    # AMR
    amr_params = params.namespace("amr")
    amr_loc = base_locator / "amr"
    amr_python_file = amr_params.existing_file("python_file")
    amr_output_dir = directory_for(amr_loc) / "documents"
    if params.string("site") == "saga":
        larger_resource = SlurmResourceRequest(memory=MemoryAmount.parse("8G"))
    else:
        larger_resource = None
    amr_job = run_python_on_args(
        amr_loc,
        amr_python_file,
        f"""
        --input {claim_detection_output.value} \
        --output {amr_output_dir} \
        --amr-parser-model {amr_params.existing_directory("model_path")}
        """,
        override_conda_config=CondaConfiguration(
            conda_base_path=params.existing_directory("conda_base_path"),
            conda_environment="transition-amr-parser"
        ),
        resource_request=larger_resource,
        depends_on=[claim_detection_output]
    )
    amr_output = ValueArtifact(
        value=amr_output_dir, depends_on=[amr_job]
    )

    # SRL
    srl_params = params.namespace("srl")
    srl_loc = base_locator / "srl"
    srl_python_file = srl_params.existing_file("python_file")
    srl_output_dir = directory_for(srl_loc) / "documents"
    srl_job = run_python_on_args(
        srl_loc,
        srl_python_file,
        f"""
        --input {amr_output.value} \
        --output {srl_output_dir} \
        --spacy-model {model_path} \
        """,
        depends_on=[amr_output]
    )
    srl_output = ValueArtifact(
        value=srl_output_dir, depends_on=[srl_job]
    )

    # Wikidata
    wikidata_params = params.namespace("wikidata")
    wikidata_loc = base_locator / "wikidata"
    wikidata_python_file = wikidata_params.existing_file("python_file")
    wikidata_output_dir = directory_for(wikidata_loc) / "documents"
    wikidata_job = run_python_on_args(
        wikidata_loc,
        wikidata_python_file,
        f"""
        --claim-input {claim_detection_output.value} \
        --srl-input {srl_output.value} \
        --amr-input {amr_output.value} \
        --output {wikidata_output_dir} \
        """,
        depends_on=[srl_output, amr_output]
    )
    wikidata_output = ValueArtifact(
        value=wikidata_output_dir, depends_on=[wikidata_job]
    )

    # Unify
    unify_params = params.namespace("unify")
    unify_loc = base_locator / "unify"
    unify_python_job = unify_params.existing_file("python_file")
    output_file = unify_params.creatable_file("output")
    run_python_on_args(
        unify_loc,
        unify_python_job,
        f"""
        --input {wikidata_output.value} \
        --output {output_file} \
        """,
        depends_on=[wikidata_output]
    )

    write_workflow_description()


if __name__ == "__main__":
    parameters_only_entry_point(main)
