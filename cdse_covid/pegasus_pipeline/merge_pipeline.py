"""Merge pipeline."""
from pegasus_wrapper import (
    Locator,
    initialize_vista_pegasus_wrapper,
    run_python_on_args,
    write_workflow_description,
)
from pegasus_wrapper.key_value import ZipKeyValueStore
from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point


def main(params: Parameters) -> None:
    """Entrypoint to merge pipeline."""
    initialize_vista_pegasus_wrapper(params)

    base_locator = Locator(("merge",))

    ingest_locator = base_locator / "ingest"
    ingest_job_file = params.existing_file("python_file")
    uiuc_aif_dir = params.existing_directory("uiuc_aif")
    name = uiuc_aif_dir.stem
    zip_dir = uiuc_aif_dir.with_suffix(".zip")
    ingest_job = run_python_on_args(
        ingest_locator / name,
        ingest_job_file,
        f"""
        --aif-dir {uiuc_aif_dir} \
        --aif-as-zip {zip_dir} \
        """,
        depends_on=[],
    )
    uiuc_store = ZipKeyValueStore(path=zip_dir, depends_on=ingest_job)

    isi_aif_dir = params.existing_directory("isi_aif")
    name = isi_aif_dir.stem
    zip_dir = isi_aif_dir.with_suffix(".zip")
    ingest_job = run_python_on_args(
        ingest_locator / name,
        ingest_job_file,
        f"""
        --aif-dir {isi_aif_dir} \
        --aif-as-zip {zip_dir} \
        """,
        depends_on=[],
    )
    isi_store = ZipKeyValueStore(path=zip_dir, depends_on=ingest_job)

    merge_job_file = params.existing_file("merge_python_file")
    output = params.creatable_directory("output")
    run_python_on_args(
        base_locator / "do_the_merge",
        merge_job_file,
        f"""
        --isi-store {isi_store.path} \
        --uiuc-store {uiuc_store.path} \
        --output {output}
        """,
        depends_on=[uiuc_store, isi_store],
    )

    write_workflow_description()


if __name__ == "__main__":
    parameters_only_entry_point(main)
