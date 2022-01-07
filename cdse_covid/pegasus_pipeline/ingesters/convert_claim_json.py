"""Convert the claim data from the output json into AIF files."""
import json
import logging
import os
from random import randint
from typing import Tuple, Any, TextIO, Dict

from aida_tools.utils import make_xml_safe, reduce_whitespace
from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

log = logging.getLogger(__name__)  # pylint:disable=invalid-name

CDSE_SYSTEM = "http://www.isi.edu/cdse"
text_to_nil_ids = {}


def write_private_data(aif_file: TextIO, private_data: str) -> None:
    """Write a component's private data."""
    aif_file.write(
        '\taida:privateData "['
        + reduce_whitespace(str(private_data)).replace('"', "")
        + ']"^^xsd:string ;\n'
    )


def write_system(aif_file: TextIO) -> None:
    """Write the `aida:system` line."""
    aif_file.write('\taida:system <' + CDSE_SYSTEM + '> .\n\n')


def generate_nil_id() -> str:
    """Return a NILX ID to use if a component lacks an entity qnode."""
    digits = []
    for _ in range(7):
        digits.append(str(randint(0, 9)))
    return "NIL" + "".join(digits)


def get_nil_id_for_entity(entity_text: str) -> str:
    """Determine the NILX ID to use for an entity."""
    nil_id = text_to_nil_ids.get(entity_text)
    if not nil_id:
        nil_id = generate_nil_id()
        text_to_nil_ids[entity_text] = nil_id
    return nil_id


def get_claim_semantics_data(
        aif_file: TextIO, data
) -> Tuple[str, str, Dict[str, Any], Dict[str, str]]:
    """Write basic claim semantics data to Claim and gather for later use."""
    claim_semantics_id = f"claim-semantics-{data['claim_semantics']['event']['qnode_id']}"
    claim_semantics = (
        "<"
        + make_xml_safe(
            CDSE_SYSTEM + "/clusters/isi/events/qnode/"
            + claim_semantics_id
        )
        + ">"
    )
    aif_file.write("\taida:claimSemantics " + claim_semantics)

    claim_arguments = {}
    argument_names_to_roles = {}
    has_claim_arguments = (
        data["claim_semantics"]["args"] is not None
    )
    if has_claim_arguments:
        argument_objects = data["claim_semantics"]["args"]
        aif_file.write(",\n")
        total_arguments = len(argument_objects)
        for arg_number, (role, arg_qnodes) in enumerate(argument_objects.items()):
            if data["claim_semantics"]["args"][f"{role}"]["type"]["qnode_id"] is not None:
                claim_argument = (
                    "<"
                    + make_xml_safe(
                        CDSE_SYSTEM + "/events/isi/qnode/EventArgument-"
                        + str(data["claim_semantics"]["args"][f"{role}"]["type"]["qnode_id"])
                    )
                    + ">"
                )
                claim_arguments[claim_argument] = data["claim_semantics"]["args"][f"{role}"]
                argument_names_to_roles[claim_argument] = role
                end_punc = " ;\n" if arg_number + 1 == total_arguments else ",\n"
                aif_file.write("\t\t" + claim_argument + end_punc)
    else:
        aif_file.write(" ;\n")
    return claim_semantics_id, claim_semantics, claim_arguments, argument_names_to_roles


def write_claim_component(
        aif_file: TextIO, data: Any, claim_component: str, claim_component_value: str
) -> None:
    """Add a claim component with the following values.
    * aida:componentName
    * aida:componentIdentity
    * aida:system
    """
    aif_file.write("aida:" + claim_component_value + " a aida:ClaimComponent ;\n")
    aif_file.write(
        '\taida:componentName "' + str(data[claim_component]) + '"^^xsd:string ;\n'
    )
    aif_file.write(
        '\taida:componentIdentity "'
        + str(data[f"{claim_component}_qnode"])
        + '"^^xsd:string .\n\n'
    )
    aif_file.write("\taida:system <" + CDSE_SYSTEM + "> .\n\n")


def write_claim_semantics_event(
        aif_file: TextIO, data: Any, claim_semantics_event: str
) -> None:
    """Add the event of the claim."""
    aif_file.write(claim_semantics_event + " a aida:Event ;\n")
    aif_file.write(
        '\taida:componentName "'
        + make_xml_safe(str(data["claim_semantics"]["event"]["text"]))
        + '"^^xsd:string ;\n'
    )
    aif_file.write(
        '\taida:componentIdentity "'
        + str(data["claim_semantics"]["event"]["qnode_id"])
        + '"^^xsd:string ;\n'
    )

    write_private_data(aif_file, data["claim_semantics"])
    write_system(aif_file)


def write_claim_semantics_argument(
        aif_file: TextIO,
        data: Any,
        source: str,
        argument: str,
        argument_role: str,
        arg_data: Any,
        claim_semantics_event: str,
        claim_semantics_id: str
) -> None:
    """Add an event argument and link it to its event."""
    aif_file.write(argument + " a aida:EventArgument ;\n")
    aif_file.write(
        '\taida:componentName "'
        + make_xml_safe(str(arg_data["type"]["text"]))
        + '"^^xsd:string ;\n'
    )
    aif_file.write(
        '\taida:componentType "'
        + str(arg_data["type"]["qnode_id"])
        + '"^^xsd:string ;\n'
    )
    defining_arg_qnode = str(arg_data["type"]["qnode_id"])
    if arg_data.get("identity") is not None and arg_data["identity"]["qnode_id"] is not None:
        aif_file.write(
            '\taida:componentIdentity "'
            + str(arg_data["identity"]["qnode_id"])
            + '"^^xsd:string ;\n'
        )
        defining_arg_qnode = arg_data["identity"]["qnode_id"]
    else:
        aif_file.write(
            '\taida:componentIdentity "'
            + get_nil_id_for_entity(arg_data["type"]["text"])
            + '"^^xsd:string ;\n'
        )

    write_private_data(aif_file, arg_data)
    write_system(aif_file)

    aif_file.write(
        "<"
        + CDSE_SYSTEM
        + "/assertions/isi/eventarg/"
        + source + "/"
        + claim_semantics_id + "/"
        + f"{data['claim_semantics']['event']['text']}.{argument_role}/"
        + f"EventArgument-{defining_arg_qnode}"
        + "> a rdf:Statement,\n"
    )
    aif_file.write('\t\taida:ArgumentStatement ;\n')
    aif_file.write(f'\trdf:object {argument} ;\n')
    aif_file.write(f'\trdf:predicate \"{argument_role}\"^^xsd:string ;\n')
    aif_file.write(f'\trdf:subject {claim_semantics_event} ;\n')
    write_system(aif_file)


def convert_json_file_to_aif(params: Parameters) -> None:
    """Convert the claim data from the output json into AIF files."""
    unify_params = params.namespace("unify")
    aif_params = params.namespace("aif")
    claims_file = unify_params.existing_file("output")
    aif_dir = aif_params.creatable_directory("aif_output_dir")
    log.info("READING: %s WRITING: %s", claims_file, aif_dir)
    cf = open(claims_file, encoding="utf-8")
    claims_data = json.load(cf)

    prior_source = ""

    af = None
    aif_file = None

    var_types_to_aida_classes = {
        "x_variable": "xVariable",
        "claimer": "claimer"
    }

    def get_name_data(data: Any, var_type: str, source: str, claim_id: str) -> Tuple[str, str, str]:
        """Get a variable's name, entity name, and justification name."
        var_type can be 'x_variable' or 'claimer'
        """
        var_count = 0
        variable_justification_name = "None"
        variable_name = (
            "<"
            + make_xml_safe(
                CDSE_SYSTEM + "/"
                + source
                + "/claim/"
                + claim_id
                + f"/{var_type}/"
                + str(var_count)
            )
            + ">"
        )
        variable_entity_name = (
            "<"
            + make_xml_safe(
                CDSE_SYSTEM + "/"
                + source
                + "/claim/"
                + claim_id
                + f"/{var_type}/entity/"
                + str(var_count)
            )
            + ">"
        )
        if (
                data[f"{var_type}_identity_qnode"] is not None
                and data[f"{var_type}_identity_qnode"]["span"] is not None
        ):
            start_offset = data[f"{var_type}_identity_qnode"]["span"][0]
            end_offset_inclusive = data[f"{var_type}_identity_qnode"]["span"][1]
            variable_justification_name = (
                "<"
                + make_xml_safe(
                    CDSE_SYSTEM
                    + source
                    + f"/{var_type}/justification"
                    + str(var_count)
                    + "/"
                    + str(start_offset)
                    + "/"
                    + str(end_offset_inclusive)
                )
                + ">"
            )
        elif data[f"{var_type}_type_qnode"]["span"] is not None:
            start_offset = data[f"{var_type}_type_qnode"]["span"][0]
            end_offset_inclusive = data[f"{var_type}_type_qnode"]["span"][1]
            variable_justification_name = (
                "<"
                + make_xml_safe(
                    CDSE_SYSTEM
                    + source
                    + f"/{var_type}/justification"
                    + str(var_count)
                    + "/"
                    + str(start_offset)
                    + "/"
                    + str(end_offset_inclusive)
                )
                + ">"
            )
        var_count += 1
        af.write(f"\taida:{var_types_to_aida_classes[var_type]} " + variable_name + " ;\n")
        associated_kes.append(variable_entity_name)
        return variable_name, variable_entity_name, variable_justification_name

    def write_qnode_data(
            data: Any,
            var_type: str,
            variable_name: str,
            variable_entity_name: str,
            variable_justification_name: str
    ) -> None:
        """Write the component, entity, and justifications for the variable."""
        af.write(variable_name + " a aida:ClaimComponent ;\n")
        af.write(
            '\taida:componentName "'
            + make_xml_safe(str(data[f"{var_type}_type_qnode"]["text"]))
            + '"^^xsd:string ;\n'
        )
        af.write(
            '\taida:componentType "'
            + str(data[f"{var_type}_type_qnode"]["qnode_id"])
            + '"^^xsd:string ;\n'
        )
        if (
                data[f"{var_type}_identity_qnode"] is not None
                and data[f"{var_type}_identity_qnode"]["qnode_id"] is not None
        ):
            variable_entity_data = data[f"{var_type}_identity_qnode"]
            variable_entity_qnode = variable_entity_data["qnode_id"]
        else:
            variable_entity_data = data[f"{var_type}_type_qnode"]
            variable_entity_qnode = get_nil_id_for_entity(variable_entity_data["text"])
        af.write(
            '\taida:componentIdentity "'
            + str(variable_entity_qnode)
            + '"^^xsd:string ;\n'
        )
        af.write("\taida:componentKE " + variable_entity_name + " ;\n")
        write_private_data(af, variable_entity_data)
        write_system(af)

        # Entity
        af.write(variable_entity_name + " a aida:Entity ;\n")
        af.write(
            '\taida:text "'
            + make_xml_safe(str(variable_entity_data["text"]))
            + '"^^xsd:string ;\n'
        )
        af.write(
            '\taida:name "'
            + str(variable_entity_qnode)
            + '"^^xsd:string ;\n'
        )
        af.write("\taida:justifiedBy " + variable_justification_name + " ;\n")
        if data[var_type]["entity"] is not None:
            write_private_data(af, data[var_type]["entity"])
        write_system(af)

        # TextJustification
        if variable_entity_data["span"] is not None:
            start_offset = variable_entity_data["span"][0]
            end_offset_inclusive = variable_entity_data["span"][1]
            af.write(variable_justification_name + " a aida:TextJustification ;\n")
            if (
                    variable_entity_data.get("confidence")
                    and variable_entity_data["confidence"] is not None
            ):
                af.write('\taida:confidence [ a aida:Confidence ;\n')
                af.write("\t\taida:confidenceValue " + f"{variable_entity_data['confidence']:.2E}" + " ;\n")
                af.write("\t\taida:system <" + CDSE_SYSTEM + "> ] ;\n")
            af.write(
                '\taida:endOffsetInclusive "' + str(end_offset_inclusive) + '"^^xsd:int ;\n'
            )
            af.write('\taida:source "' + source + '"^^xsd:string ;\n')
            af.write('\taida:startOffset "' + str(start_offset) + '"^^xsd:int ;\n')
            write_system(af)

    for data in claims_data:
        source = os.path.splitext(str(data["doc_id"]))[0]

        if source != prior_source:
            if "" != prior_source:
                af.close()  # type: ignore
            aif_file = os.path.join(aif_dir, source + ".ttl")
            af = open(aif_file, "w", encoding="utf-8")
            log.info("WRITING: %s", aif_file)
            af.write(
                "@prefix aida: <https://raw.githubusercontent.com/NextCenturyCorporation/AIDA-Interchange-Format/develop/java/src/main/resources/com/ncc/aif/aida_ontology.shacl> .\n"
            )
            af.write("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n")
            af.write("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n\n")
            prior_source = source

        if af:
            # Write the Claim
            claim_id = make_xml_safe(str(data["claim_id"]))
            claim_name = (
                "<"
                + make_xml_safe(CDSE_SYSTEM + source + "/claim_id/" + claim_id)
                + ">"
            )
            af.write(claim_name + " a aida:Claim ;\n")
            af.write('\taida:sourceDocument "' + str(data["doc_id"]) + '"^^xsd:string ;\n')
            claim_id = make_xml_safe(str(data["claim_id"]))
            af.write('\taida:claimId "' + claim_id + '"^^xsd:string ;\n')
            af.write('\taida:topic "' + str(data["topic"]) + '"^^xsd:string ;\n')
            af.write('\taida:subtopic "' + str(data["subtopic"]) + '"^^xsd:string ;\n')
            af.write('\taida:claimTemplate "' + str(data["claim_template"]) + '"^^xsd:string ;\n')

            has_x_variable = data["x_variable_type_qnode"] is not None
            x_variable_name = "None"
            x_variable_entity_name = "None"
            x_variable_justification_name = "None"

            associated_kes = []

            if has_x_variable:
                x_variable_name, x_variable_entity_name, x_variable_justification_name = (
                    get_name_data(data, "x_variable", source, claim_id)
                )

            af.write(
                '\taida:naturalLanguageDescription "'
                + reduce_whitespace(str(data["claim_sentence"])).replace('"', "")
                + '"^^xsd:string ;\n'
            )

            has_claim_semantics = (
                data["claim_semantics"] is not None
                and data["claim_semantics"]["event"] is not None
                and data["claim_semantics"]["event"]["qnode_id"] is not None
            )
            claim_semantics = "None"
            claim_semantics_id = "None"
            claim_arguments = {}
            argument_names_to_roles = {}

            if has_claim_semantics:
                claim_semantics_id, claim_semantics, claim_arguments, argument_names_to_roles = (
                    get_claim_semantics_data(af, data)
                )
                associated_kes.append(claim_semantics)
                associated_kes.extend([claim_arg for claim_arg in claim_arguments])

            has_claimer = data["claimer_type_qnode"] is not None
            claimer_name = "None"
            claimer_entity_name = "None"
            claimer_justification_name = "None"

            if has_claimer:
                claimer_name, claimer_entity_name, claimer_justification_name = (
                    get_name_data(data, "claimer", source, claim_id)
                )

            has_claim_location = data["claim_location_qnode"] is not None
            claim_location = "None"
            if has_claim_location:
                claim_location = make_xml_safe(data["claim_location_qnode"])
                af.write("\taida:claimLocation aida:" + claim_location + " ;\n")

            start_offset = data["claim_span"][0]
            end_offset_inclusive = data["claim_span"][1]
            claim_justification_name = (
                "<"
                + make_xml_safe(
                    CDSE_SYSTEM + "/claims/justifications/"
                    + source
                    + "/"
                    + str(start_offset)
                    + "/"
                    + str(end_offset_inclusive)
                )
                + ">"
            )

            if associated_kes:
                af.write('\taida:associatedKEs ')
                len_associated_kes = len(associated_kes)
                for i, ke in enumerate(associated_kes):
                    end_punc = " ;\n" if i + 1 == len_associated_kes else ",\n"
                    if i == 0:
                        af.write(ke + end_punc)
                    else:
                        af.write('\t\t' + ke + end_punc)

            af.write("\taida:justifiedBy " + claim_justification_name + " ;\n")
            af.write('\taida:claimText "' + str(data["claim_text"]) + '"^^xsd:string ;\n')
            write_system(af)

            # Write each component
            af.write(claim_justification_name + " a aida:TextJustification ;\n")
            af.write('\taida:endOffsetInclusive "' + str(end_offset_inclusive) + '"^^xsd:int ;\n')
            af.write('\taida:source "' + source + '"^^xsd:string ;\n')
            af.write('\taida:startOffset "' + str(start_offset) + '"^^xsd:int ;\n')
            af.write(f"\taida:system <{CDSE_SYSTEM}> . \n\n")

            if has_x_variable:
                write_qnode_data(
                    data,
                    "x_variable",
                    x_variable_name,
                    x_variable_entity_name,
                    x_variable_justification_name
                )

            if has_claimer:
                write_qnode_data(
                    data,
                    "claimer",
                    claimer_name,
                    claimer_entity_name,
                    claimer_justification_name
                )

            if has_claim_location:
                write_claim_component(af, data, "claim_location", claim_location)

            if has_claim_semantics:
                # SameAsCluster
                af.write(claim_semantics + " a aida:SameAsCluster ;\n")
                claim_semantics_event = (
                    "<"
                    + make_xml_safe(
                        CDSE_SYSTEM
                        + "/events/isi/qnode/claim-semantics-"
                        + str(data["claim_semantics"]["event"]["qnode_id"])
                    )
                    + ">"
                )
                af.write('\taida:prototype ' + claim_semantics_event + ' ;\n')
                af.write("\taida:system <" + CDSE_SYSTEM + "> .\n\n")

                # Event
                write_claim_semantics_event(af, data, claim_semantics_event)

                # Arguments
                for argument, arg_data in claim_arguments.items():
                    argument_role = argument_names_to_roles[argument]

                    write_claim_semantics_argument(
                        af,
                        data,
                        source,
                        argument,
                        argument_role,
                        arg_data,
                        claim_semantics_event,
                        claim_semantics_id
                    )

    if af:
        af.close()
        log.info("WROTE: %s", aif_file)


if __name__ == "__main__":
    parameters_only_entry_point(convert_json_file_to_aif)
