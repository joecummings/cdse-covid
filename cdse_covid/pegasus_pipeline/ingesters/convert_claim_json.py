"""Convert the claim data from the output json into AIF files."""
import json
import logging
import os
from random import randint
from typing import Any, Dict, TextIO, Tuple, Union

from aida_tools.utils import make_xml_safe, reduce_whitespace
from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

log = logging.getLogger(__name__)  # pylint:disable=invalid-name

CDSE_SYSTEM = "http://www.isi.edu/cdse"
text_to_nil_ids: Dict[str, str] = {}


def write_private_data(aif_file: TextIO, private_data: str) -> None:
    """Write a component's private data."""
    aif_file.write(
        "\taida:privateData [ a aida:PrivateData ;\n"
        + '\t\taida:jsonContent "'
        + reduce_whitespace(str(private_data)).replace('"', "")
        + '"^^xsd:string ;\n'
        + "\t\taida:system <"
        + CDSE_SYSTEM
        + "> ] ;\n"
    )


def write_system(aif_file: TextIO) -> None:
    """Write the `aida:system` line that concludes each component."""
    aif_file.write("\taida:system <" + CDSE_SYSTEM + "> .\n\n")


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


def write_entity_data(
    aif_file: TextIO,
    source: str,
    var_type: str,
    entity_name: str,
    entity_data: Any,
    var_count: Union[int, str],
) -> None:
    """Write the entity data."""
    justification_name = None
    start_offset = None
    end_offset_inclusive = None
    if entity_data.get("span") is not None:
        start_offset = entity_data["span"][0]
        end_offset_inclusive = entity_data["span"][1]
        justification_name = (
            "<"
            + make_xml_safe(
                CDSE_SYSTEM
                + "/"
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

    # Entity
    aif_file.write(entity_name + " a aida:Entity ;\n")
    aif_file.write('\taida:text "' + make_xml_safe(str(entity_data["text"])) + '"^^xsd:string ;\n')
    if justification_name:
        aif_file.write("\taida:justifiedBy " + str(justification_name) + " ;\n")
    if entity_data["entity"] is not None:
        write_private_data(aif_file, entity_data["entity"])
    write_system(aif_file)

    # TextJustification
    if justification_name:
        aif_file.write(justification_name + " a aida:TextJustification ;\n")
        confidence_val = (
            entity_data["confidence"]
            if entity_data.get("confidence") and entity_data["confidence"] is not None
            else 1.0
        )
        aif_file.write("\taida:confidence [ a aida:Confidence ;\n")
        aif_file.write("\t\taida:confidenceValue " + f"{confidence_val:.2E}" + " ;\n")
        aif_file.write("\t\taida:system <" + CDSE_SYSTEM + "> ] ;\n")
        aif_file.write('\taida:endOffsetInclusive "' + str(end_offset_inclusive) + '"^^xsd:int ;\n')
        aif_file.write('\taida:source "' + source + '"^^xsd:string ;\n')
        aif_file.write('\taida:startOffset "' + str(start_offset) + '"^^xsd:int ;\n')
        write_system(aif_file)


def get_claim_semantics_data(
    aif_file: TextIO, source: str, data: Any, event_number: int
) -> Tuple[str, str, Dict[str, Any], Dict[str, str]]:
    """Write basic claim semantics data to Claim and gather for later use."""
    event_number_string = str(event_number).zfill(6)
    claim_semantics_event_id = f"EN_Event_{event_number_string}"
    claim_semantics = (
        "<" + make_xml_safe(CDSE_SYSTEM + f"/clusters/isi/events/{source}/{claim_semantics_event_id}") + ">"
    )
    aif_file.write("\taida:claimSemantics " + claim_semantics)

    claim_arguments = {}
    argument_names_to_roles = {}
    has_claim_arguments = data["claim_semantics"]["args"] is not None
    has_arg_qnodes = False
    arg_count = 0
    if has_claim_arguments:
        argument_objects = data["claim_semantics"]["args"]
        total_arguments = len(argument_objects)
        for arg_number, (role, arg_qnodes) in enumerate(argument_objects.items()):
            claim_argument = None
            argument_id = "EN_Entity_EDL_ENG_" + str(arg_count).zfill(7)
            if arg_qnodes.get("type"):
                if arg_qnodes["type"]["qnode_id"] is not None:
                    claim_argument = (
                        "<"
                        + make_xml_safe(
                            CDSE_SYSTEM
                            + "/clusters/isi/entity/"
                            + source + "/"
                            + argument_id
                        )
                        + ">"
                    )
            elif arg_qnodes.get("identity"):
                if arg_qnodes["identity"]["qnode_id"] is not None:
                    claim_argument = (
                        "<"
                        + make_xml_safe(
                            CDSE_SYSTEM
                            + "/clusters/isi/entity/"
                            + source + "/"
                            + argument_id
                        )
                        + ">"
                    )
            if claim_argument:
                has_arg_qnodes = True
                claim_arguments[claim_argument] = arg_qnodes
                argument_names_to_roles[claim_argument] = role
                end_punc = " ;\n" if arg_number + 1 == total_arguments else ""
                aif_file.write(",\n\t\t" + claim_argument + end_punc)
                arg_count += 1
            elif arg_number + 1 == total_arguments:
                aif_file.write(" ;\n")

    if not has_arg_qnodes:
        aif_file.write(" ;\n")
    return claim_semantics_event_id, claim_semantics, claim_arguments, argument_names_to_roles


def write_claim_component(
    aif_file: TextIO, data: Any, claim_component: str, claim_component_value: str
) -> None:
    """Add a claim component with the following values.

    * aida:componentName
    * aida:componentIdentity
    * aida:system
    """
    aif_file.write("aida:" + claim_component_value + " a aida:ClaimComponent ;\n")
    aif_file.write('\taida:componentName "' + str(data[claim_component]) + '"^^xsd:string ;\n')
    aif_file.write(
        '\taida:componentIdentity "' + str(data[f"{claim_component}_qnode"]) + '"^^xsd:string .\n\n'
    )
    aif_file.write("\taida:system <" + CDSE_SYSTEM + "> .\n\n")


def write_claim_semantics_event(
    aif_file: TextIO, source: str, data: Any, claim_semantics_event: str, event_id: str
) -> None:
    """Add the event of the claim and its justifications."""
    aif_file.write(claim_semantics_event + " a aida:Event ;\n")
    event_data = data["claim_semantics"]["event"]

    confidence_val = (
        event_data["confidence"]
        if event_data.get("confidence") and event_data["confidence"] is not None
        else 1.0
    )
    confidence_aif = (
        "\taida:confidence [ a aida:Confidence ;\n"
        + "\t\taida:confidenceValue "
        + f"{confidence_val:.2E}"
        + " ;\n"
        + "\t\taida:system <"
        + CDSE_SYSTEM
        + "> ] ;\n"
    )
    aif_file.write(confidence_aif)

    # informative justification
    # events have no spans in our system
    informative_justification_name = (
        "<"
        + CDSE_SYSTEM
        + f"/assertions/isi/event_informative_justification/{event_id}/"
        + source
        + ">"
    )
    aif_file.write(f"\taida:informativeJustification {informative_justification_name} ;\n")

    # justified by
    justification_name = (
        "<"
        + CDSE_SYSTEM
        + f"/assertions/isi/eventjustification/{event_id}/{source}"
        + ">"
    )
    aif_file.write(f"\taida:justifiedBy {justification_name} ;\n")

    write_private_data(aif_file, data["claim_semantics"])
    write_system(aif_file)

    # TextJustification
    aif_file.write(informative_justification_name + " a aida:TextJustification ;\n")
    aif_file.write(confidence_aif)
    write_private_data(aif_file, event_data)
    aif_file.write(f'\taida:source "{source}"^^xsd:string ;\n')
    write_system(aif_file)


def write_claim_semantics_argument(
    aif_file: TextIO,
    data: Any,
    source: str,
    argument: str,
    argument_role: str,
    arg_data: Any,
    claim_semantics_event: str,
    claim_semantics_id: str,
    arg_count: Union[int, str],
) -> None:
    """Add an event argument and link it to its event."""
    # Check if there's type/identity data
    # Most args should have a type
    arg_type_data = arg_data.get("type")
    arg_identity_data = arg_data.get("identity")
    if not arg_type_data:
        logging.warning("No argument type data in %s", arg_data)
        if not arg_identity_data:
            return
    defining_arg_qnode = ""
    defining_arg_data = {}
    # Writing entity fields
    if arg_type_data:
        defining_arg_data = arg_type_data
        defining_arg_qnode = str(arg_type_data["qnode_id"])
    if arg_identity_data and arg_identity_data["qnode_id"] is not None:
        defining_arg_qnode = arg_identity_data["qnode_id"]
        defining_arg_data = arg_identity_data

    write_entity_data(
        aif_file,
        source,
        argument_role,
        argument,
        defining_arg_data,
        arg_count,
    )

    aif_file.write(
        "<"
        + CDSE_SYSTEM
        + "/assertions/isi/eventarg/"
        + source
        + "/"
        + claim_semantics_id
        + "/"
        + f"{data['claim_semantics']['event']['text']}.{argument_role}/"
        + f"EventArgument-{defining_arg_qnode}"
        + "> a rdf:Statement,\n"
    )
    aif_file.write("\t\taida:ArgumentStatement ;\n")
    aif_file.write(f"\trdf:object {argument} ;\n")
    aif_file.write(f'\trdf:predicate "{argument_role}"^^xsd:string ;\n')
    aif_file.write(f"\trdf:subject {claim_semantics_event} ;\n")
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

    var_types_to_aida_classes = {"x_variable": "xVariable", "claimer": "claimer"}

    def get_name_data(
        aida_file: TextIO, var_type: str, source: str, claim_id: str
    ) -> Tuple[str, str]:
        """Get a variable's name, entity name, and justification name.

        var_type can be 'x_variable' or 'claimer'
        """
        var_count = 0
        # variable_name = (
        #     "<"
        #     + make_xml_safe(
        #         CDSE_SYSTEM + "/" + source + "/claim/" + claim_id + f"/{var_type}/" + str(var_count)
        #     )
        #     + ">"
        # )

        type_for_uri = "X" if var_type == "x_variable" else var_type
        variable_name = (
            f"ex:claim_{source}_{claim_id}_{type_for_uri}"
        )
        variable_entity_name = (
            "<"
            + make_xml_safe(
                CDSE_SYSTEM
                + "/"
                + source
                + "/claim/"
                + claim_id
                + f"/{var_type}/entity/"
                + str(var_count)
            )
            + ">"
        )
        var_count += 1
        aida_file.write(f"\taida:{var_types_to_aida_classes[var_type]} " + variable_name + " ;\n")
        associated_kes.append(variable_entity_name)
        return variable_name, variable_entity_name

    def write_qnode_data(
        aif_file: TextIO,
        data: Any,
        var_type: str,
        variable_name: str,
        variable_entity_name: str,
        var_count: int,
    ) -> None:
        """Write the component, entity, and justifications for the variable."""
        if (
            data[f"{var_type}_identity_qnode"] is not None
            and data[f"{var_type}_identity_qnode"]["qnode_id"] is not None
        ):
            variable_entity_data = data[f"{var_type}_identity_qnode"]
            variable_entity_qnode = variable_entity_data["qnode_id"]
        else:
            variable_entity_data = data[f"{var_type}_type_qnode"]
            variable_entity_qnode = get_nil_id_for_entity(variable_entity_data["text"])
        aif_file.write(variable_name + " a aida:ClaimComponent ;\n")
        aif_file.write(
            '\taida:componentName "'
            + make_xml_safe(str(variable_entity_data["text"]))
            + '"^^xsd:string ;\n'
        )
        aif_file.write(
            '\taida:componentType "'
            + str(data[f"{var_type}_type_qnode"]["qnode_id"])
            + '"^^xsd:string ;\n'
        )
        aif_file.write(
            '\taida:componentIdentity "' + str(variable_entity_qnode) + '"^^xsd:string ;\n'
        )
        aif_file.write("\taida:componentKE " + variable_entity_name + " ;\n")
        write_private_data(aif_file, variable_entity_data)
        write_system(aif_file)

        # Entity and justification
        write_entity_data(
            aif_file,
            source,
            var_type,
            variable_entity_name,
            variable_entity_data,
            var_count,
        )

    event_count = 0

    for data in claims_data:
        source = os.path.splitext(str(data["doc_id"]))[0]

        if source != prior_source:
            if "" != prior_source:
                af.close()  # type: ignore
            aif_file = os.path.join(aif_dir, source + ".ttl")
            af = open(aif_file, "w", encoding="utf-8")
            log.info("WRITING: %s", aif_file)
            af.write(
                "@prefix aida: <https://raw.githubusercontent.com/NextCenturyCorporation/AIDA-Interchange-Format/master/java/src/main/resources/com/ncc/aif/ontologies/InterchangeOntology#> .\n"
            )
            af.write("@prefix ex: <https://www.caci.com/claim-example#> .\n")
            af.write("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n")
            af.write("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n\n")
            af.write(f"<{CDSE_SYSTEM}> a aida:System .\n\n")
            prior_source = source

        if af:
            # Write the Claim
            claim_id = make_xml_safe(str(data["claim_id"]))
            claim_name = "<" + make_xml_safe(CDSE_SYSTEM + "/" + source + "/claim_id/" + claim_id) + ">"
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

            associated_kes = []

            if has_x_variable:
                (
                    x_variable_name,
                    x_variable_entity_name,
                ) = get_name_data(af, "x_variable", source, claim_id)

            af.write(
                '\taida:naturalLanguageDescription "'
                + reduce_whitespace(str(data["claim_text"])).replace('"', "")
                + '"^^xsd:string ;\n'
            )

            has_claim_semantics = (
                data["claim_semantics"] is not None
                and data["claim_semantics"]["event"] is not None
                and data["claim_semantics"]["event"]["qnode_id"] is not None
                and data["claim_semantics"]["event"]["text"] is not "Unspecified"
            )
            claim_semantics_cluster_name = "None"
            claim_semantics_id = "None"
            claim_arguments: Dict[str, Any] = {}
            argument_names_to_roles: Dict[str, str] = {}

            if has_claim_semantics:
                (
                    claim_semantics_id,
                    claim_semantics_cluster_name,
                    claim_arguments,
                    argument_names_to_roles,
                ) = get_claim_semantics_data(af, source, data, event_count)
                event_count += 1
                associated_kes.append(claim_semantics_cluster_name)
                associated_kes.extend([claim_arg for claim_arg in claim_arguments])

            has_claimer = data["claimer_type_qnode"] is not None
            claimer_name = "None"
            claimer_entity_name = "None"

            if has_claimer:
                claimer_name, claimer_entity_name = get_name_data(af, "claimer", source, claim_id)

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
                    CDSE_SYSTEM
                    + "/claims/justifications/"
                    + source
                    + "/"
                    + str(start_offset)
                    + "/"
                    + str(end_offset_inclusive)
                )
                + ">"
            )

            if associated_kes:
                af.write("\taida:associatedKEs ")
                len_associated_kes = len(associated_kes)
                for i, ke in enumerate(associated_kes):
                    end_punc = " ;\n" if i + 1 == len_associated_kes else ",\n"
                    if i == 0:
                        af.write(ke + end_punc)
                    else:
                        af.write("\t\t" + ke + end_punc)

            af.write("\taida:justifiedBy " + claim_justification_name + " ;\n")
            write_system(af)

            # Write each component
            af.write(claim_justification_name + " a aida:TextJustification ;\n")
            af.write("\taida:confidence [ a aida:Confidence ;\n")
            af.write("\t\taida:confidenceValue " + f"{1.00:.2E}" + " ;\n")
            af.write("\t\taida:system <" + CDSE_SYSTEM + "> ] ;\n")
            af.write('\taida:endOffsetInclusive "' + str(end_offset_inclusive) + '"^^xsd:int ;\n')
            af.write('\taida:source "' + source + '"^^xsd:string ;\n')
            af.write('\taida:startOffset "' + str(start_offset) + '"^^xsd:int ;\n')
            af.write(f"\taida:system <{CDSE_SYSTEM}> . \n\n")

            if has_x_variable:
                write_qnode_data(
                    af,
                    data,
                    "x_variable",
                    x_variable_name,
                    x_variable_entity_name,
                    0,
                )

            if has_claimer:
                write_qnode_data(
                    af,
                    data,
                    "claimer",
                    claimer_name,
                    claimer_entity_name,
                    0,
                )

            if has_claim_location:
                write_claim_component(af, data, "claim_location", claim_location)

            if has_claim_semantics:
                # SameAsCluster
                af.write(claim_semantics_cluster_name + " a aida:SameAsCluster ;\n")
                claim_semantics_event_name = (
                    "<"
                    + CDSE_SYSTEM
                    + f"/events/isi/{source}/{claim_semantics_id}"
                    + ">"
                )
                af.write("\taida:prototype " + claim_semantics_event_name + " ;\n")
                write_system(af)

                # Event
                write_claim_semantics_event(af, source, data, claim_semantics_event_name, claim_semantics_id)

                # Argument SameAsClusters
                clusters_to_arg_entities = {}
                for argument in claim_arguments.keys():
                    arg_id = argument.split("/")[-1].strip(">")
                    af.write(argument + "a aida:SameAsCluster ;\n")
                    claim_semantics_arg_name = (
                        "<"
                        + CDSE_SYSTEM
                        + f"/entities/isi/{source}/{arg_id}"
                        + ">"
                    )
                    af.write("\taida:prototype " + claim_semantics_arg_name + " ;\n")
                    write_system(af)
                    clusters_to_arg_entities[argument] = claim_semantics_arg_name

                # Arguments
                for argument, arg_data in claim_arguments.items():
                    argument_role = argument_names_to_roles[argument]
                    arg_name = clusters_to_arg_entities[argument]
                    arg_number = arg_name.split("_")[-1].strip(">")

                    print(f"DEBUGGING! arg_data = {arg_data}")

                    write_claim_semantics_argument(
                        af,
                        data,
                        source,
                        arg_name,
                        argument_role,
                        arg_data,
                        claim_semantics_event_name,
                        claim_semantics_id,
                        arg_number,
                    )

    if af:
        af.close()
        log.info("WROTE: %s", aif_file)


if __name__ == "__main__":
    parameters_only_entry_point(convert_json_file_to_aif)
