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

    # AMR parsing over the entirety of each document
    amr_params = params.namespace("amr")
    amr_all_loc = base_locator / "amr_all"
    amr_all_python_file = amr_params.existing_file("python_file_all")
    amr_all_output_dir = directory_for(amr_all_loc) / "documents"
    if params.string("site") == "saga":
        larger_resource = SlurmResourceRequest(memory=MemoryAmount.parse("8G"))
    else:
        larger_resource = None
    amr_all_job = run_python_on_args(
        amr_all_loc,
        amr_all_python_file,
        f"""
        --corpus {input_corpus_dir}
        --output {amr_all_output_dir}
        --amr-parser-model {amr_params.existing_directory("model_path")}
        """,
        override_conda_config=CondaConfiguration(
            conda_base_path=params.existing_directory("conda_base_path"),
            conda_environment="transition-amr-parser"
        ),
        resource_request=larger_resource,
        depends_on=[]
    )
    amr_all_output = ValueArtifact(
        value=amr_all_output_dir, depends_on=[amr_all_job]
    )

    # Find claims
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

    # Find x variable
    x_var_params = params.namespace("x_variable")
    x_var_job_loc = base_locator / "x_variable"
    x_var_python_file = x_var_params.existing_file("python_file")
    x_var_output_dir = directory_for(x_var_job_loc) / "documents"
    x_variable_job = run_python_on_args(
        x_var_job_loc,
        x_var_python_file,
        f"""
        --input {claim_detection_output.value} \
        --output {x_var_output_dir} \
        --amr-parser-model {amr_params.existing_directory("model_path")} \
        --domain {amr_params.string("domain")}
        """,
        depends_on=[claim_detection_output]
    )
    x_var_output = ValueArtifact(
        value=x_var_output_dir, depends_on=[x_variable_job]
    )

    # Find claimer
    claimer_params = params.namespace("claimer")
    claimer_job_loc = base_locator / "claimer"
    claimer_python_file = claimer_params.existing_file("python_file")
    claimer_output_dir = directory_for(claimer_job_loc) / "documents"
    claimer_job = run_python_on_args(
        claimer_job_loc,
        claimer_python_file,
        f"""
        --claim-input {claim_detection_output.value} \
        --ouptut {claimer_output_dir} \
        --amr-model $PATH
        """,
        depends_on=[claim_detection_output]
    )
    claimer_output = ValueArtifact(
        value=claimer_output_dir, depends_on=[claimer_job]
    )
    
    # Link DWD for important objects
    wikidata_params = params.namespace("wikidata")
    wikidata_job_loc = base_locator / "wikidata"
    wikidata_python_file = wikidata_params.existing_file("python_file")
    wikidata_output_dir = directory_for(wikidata_job_loc) / "documents"
    wikidata_job = run_python_on_args(
        wikidata_job_loc,
        wikidata_python_file,
        f"""
        --input {claim_detection_output.value} {x_var_output.value} {claimer_output.value} \
        --output {wikidata_output_dir} \
        --use-overlay
        """
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
