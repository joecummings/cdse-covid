from pegasus_wrapper import (
    initialize_vista_pegasus_wrapper,
    run_python_on_parameters,
    write_workflow_description,
)
from pegasus_wrapper.locator import Locator
from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point
from cdse_covid.pegasus_pipeline import run_claim_detection


def main(params: Parameters):
    initialize_vista_pegasus_wrapper(params)

    run_python_on_parameters(
        Locator(("claim_detection",)),
        run_claim_detection,
        params.namespace("claim_detection"),
        depends_on=[],
    )

    write_workflow_description()


if __name__ == "__main__":
    parameters_only_entry_point(main)
