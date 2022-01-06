"""Convert the claim data from the output json into AIF files."""
import json
import logging
import os
from random import randint
from typing import Tuple, Any

from aida_tools.utils import make_xml_safe, reduce_whitespace
from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

log = logging.getLogger(__name__)  # pylint:disable=invalid-name


def convert_json_file_to_aif(params: Parameters) -> None:
    """Convert the claim data from the output json into AIF files."""
    unify_params = params.namespace("unify")
    aif_params = params.namespace("aif")
    claims_file = unify_params.existing_file("output")
    aif_dir = aif_params.creatable_directory("aif_output_dir")
    log.info("READING: %s WRITING: %s", claims_file, aif_dir)
    cf = open(claims_file, encoding="utf-8")
    claims_data = json.load(cf)

    cdse_system = "http://www.isi.edu/cdse"

    prior_source = ""

    af = None
    aif_file = None

    var_types_to_aida_classes = {
        "x_variable": "xVariable",
        "claimer": "claimer"
    }

    text_to_nil_ids = {}

    def get_name_data(data: Any, var_type: str, source: str, claim_id: str) -> Tuple[str, str, str]:
        """Get a variable's name, entity name, and justification name."
        var_type can be 'x_variable' or 'claimer'
        """
        var_count = 0
        variable_justification_name = "None"
        variable_name = (
            "<"
            + make_xml_safe(
                cdse_system + "/"
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
                cdse_system + "/"
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
                    cdse_system
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
                    cdse_system
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

    def write_system() -> None:
        """Write the `aida:system` line."""
        af.write('\taida:system <' + cdse_system + '> .\n\n')

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
        af.write(
            '\taida:privateData "['
            + reduce_whitespace(str(variable_entity_data))
            + ']"^^xsd:string ;\n'
        )
        write_system()

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
            af.write(
                '\taida:privateData "['
                + reduce_whitespace(str(data[var_type]["entity"]))
                + ']"^^xsd:string ;\n'
            )
        write_system()

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
                af.write("\t\taida:system <" + cdse_system + "> ] ;\n")
            af.write(
                '\taida:endOffsetInclusive "' + str(end_offset_inclusive) + '"^^xsd:int ;\n'
            )
            af.write('\taida:source "' + source + '"^^xsd:string ;\n')
            af.write('\taida:startOffset "' + str(start_offset) + '"^^xsd:int ;\n')
            write_system()

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
            claim_id = make_xml_safe(str(data["claim_id"]))
            claim_name = (
                "<"
                + make_xml_safe(cdse_system + source + "/claim_id/" + claim_id)
                + ">"
            )
            af.write(claim_name + " a aida:Claim ;\n")
            af.write('\taida:sourceDocument "' + str(data["doc_id"]) + '"^^xsd:string ;\n')
            claim_id = make_xml_safe(str(data["claim_id"]))
            af.write('\taida:claimId "' + claim_id + '"^^xsd:string ;\n')
            # Not supported: queryId
            #      af.write('\taida:queryId "None" ;\n')
            # Not supported: importance
            #      af.write('\taida:importance "0.0"^^xsd:double ;\n')
            af.write('\taida:topic "' + str(data["topic"]) + '"^^xsd:string ;\n')
            af.write('\taida:subtopic "' + str(data["subtopic"]) + '"^^xsd:string ;\n')
            af.write('\taida:claimTemplate "' + str(data["claim_template"]) + '"^^xsd:string ;\n')

            associated_kes = []

            has_x_variable = data["x_variable_type_qnode"] is not None
            x_variable_name = "None"
            x_variable_entity_name = "None"
            x_variable_justification_name = "None"

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
            # alternative key check syntax: 'qnode_id' in data['claim_semantics']['event']
            claim_semantics = "None"
            claim_semantics_id = "None"
            claim_arguments = {}
            argument_names_to_roles = {}
            if has_claim_semantics:
                claim_semantics_id = f"claim-semantics-{data['claim_semantics']['event']['qnode_id']}"
                claim_semantics = (
                    "<"
                    + make_xml_safe(
                        cdse_system + "/clusters/isi/events/qnode/"
                        + claim_semantics_id
                    )
                    + ">"
                )
                af.write("\taida:claimSemantics " + claim_semantics)
                associated_kes.append(claim_semantics)

                has_claim_arguments = (
                    data["claim_semantics"]["args"] is not None
                )
                if has_claim_arguments:
                    argument_objects = data["claim_semantics"]["args"]
                    af.write(",\n")
                    total_arguments = len(argument_objects)
                    for arg_number, (role, arg_qnodes) in enumerate(argument_objects.items()):
                        if data["claim_semantics"]["args"][f"{role}"]["type"]["qnode_id"] is not None:
                            claim_argument = (
                                "<"
                                + make_xml_safe(
                                    cdse_system + "/events/isi/qnode/EventArgument-"
                                    + str(data["claim_semantics"]["args"][f"{role}"]["type"]["qnode_id"])
                                )
                                + ">"
                            )
                            claim_arguments[claim_argument] = data["claim_semantics"]["args"][f"{role}"]
                            argument_names_to_roles[claim_argument] = role
                            associated_kes.append(claim_argument)
                            end_punc = " ;\n" if arg_number + 1 == total_arguments else ",\n"
                            af.write("\t\t" + claim_argument + end_punc)
                else:
                    af.write(" ;\n")

            has_claimer = data["claimer_type_qnode"] is not None
            claimer_name = "None"
            claimer_entity_name = "None"
            claimer_justification_name = "None"

            if has_claimer:
                claimer_name, claimer_entity_name, claimer_justification_name = (
                    get_name_data(data, "claimer", source, claim_id)
                )

            # Not supported optional: claimerAffiliation
            # TODO: check epistemic status is one of:
            # True_Certain, True_Uncertain, False_Certain, False_Uncertain, Unknown
            # STRING?      af.write('\taida:epistemic aida:' + '"Unknown"^^xsd:string' + ' ;\n')
            # TODO: check sentiment status is one of:
            # Positive, Negative, Mixed, Neutral_Unknown
            # STRING?      af.write('\taida:sentiment aida:' + '"NeutralUnknown"^^xsd:string' + ' ;\n')

            has_claim_date_time = data["claim_date_time"] is not None
            if has_claim_date_time:
                # TODO: no claim_date_time values have been sent, they will need to be converted to LDCTime
                af.write("\taida:claimDateTime aida:TimeTBD ;\n")

            has_claim_location = data["claim_location_qnode"] is not None
            claim_location = "None"
            if has_claim_location:
                claim_location = make_xml_safe(data["claim_location_qnode"])
                af.write("\taida:claimLocation aida:" + claim_location + " ;\n")
            # Not supported: associatedKEs, identicalClaims, relatedClaims, supportingClaims, refutingClaims

            start_offset = data["claim_span"][0]
            end_offset_inclusive = data["claim_span"][1]
            claim_justification_name = (
                "<"
                + make_xml_safe(
                    cdse_system + "/claims/justifications/"
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
            # NOTE: claimText is not in AIF spec, but included for completeness
            af.write('\taida:claimText "' + str(data["claim_text"]) + '"^^xsd:string ;\n')
            write_system()

            af.write(claim_justification_name + " a aida:TextJustification ;\n")
            af.write('\taida:endOffsetInclusive "' + str(end_offset_inclusive) + '"^^xsd:int ;\n')
            af.write('\taida:source "' + source + '"^^xsd:string ;\n')
            af.write('\taida:startOffset "' + str(start_offset) + '"^^xsd:int ;\n')
            af.write(f"\taida:system <{cdse_system}> . \n\n")

            if has_x_variable:
                write_qnode_data(
                    data,
                    "x_variable",
                    x_variable_name,
                    x_variable_entity_name,
                    x_variable_justification_name
                )

            # <http://www.isi.edu/gaia/entities/uiuc/L0C04959A/L0C04959A-author> a aida:Entity ;
            #    aida:confidence [ a aida:Confidence ;
            #            aida:confidenceValue 1e+00 ;
            #            aida:system <http://www.uiuc.edu> ] ;
            #    aida:link [ a aida:LinkAssertion ;
            #            aida:confidence [ a aida:Confidence ;
            #                    aida:confidenceValue 1e+00 ;
            #                    aida:system <http://www.uiuc.edu> ] ;
            #            aida:linkTarget "NIL000000055"^^xsd:string ;
            #            aida:system <http://www.uiuc.edu> ] ;
            #    aida:system <https://www.isi.edu> .

            #    "x_variable": {
            #      "text": "transmissible infectious disease",
            #      "mention_id": null,
            #      "entity": { "ent_id": "0016164", "ent_type": "MHI" },
            #      "doc_id": "L0C04ATRE.rsd",
            #      "span": [1744, 1776]

            if has_claimer:
                write_qnode_data(
                    data,
                    "claimer",
                    claimer_name,
                    claimer_entity_name,
                    claimer_justification_name
                )

            if has_claim_location:
                af.write("aida:" + claim_location + " a aida:ClaimComponent ;\n")
                af.write(
                    '\taida:componentName "' + str(data["claim_location"]) + '"^^xsd:string ;\n'
                )
                af.write(
                    '\taida:componentIdentity "'
                    + str(data["claim_location_qnode"])
                    + '"^^xsd:string .\n\n'
                )
                af.write("\taida:system <" + cdse_system + "> .\n\n")

            if has_claim_semantics:
                # SameAsCluster
                af.write(claim_semantics + " a aida:SameAsCluster ;\n")
                claim_semantics_event = (
                    "<"
                    + make_xml_safe(
                        cdse_system
                        + "/events/isi/qnode/claim-semantics-"
                        + str(data["claim_semantics"]["event"]["qnode_id"])
                    )
                    + ">"
                )
                af.write('\taida:prototype ' + claim_semantics_event + ' ;\n')
                af.write("\taida:system <" + cdse_system + "> .\n\n")

                # Event
                af.write(claim_semantics_event + " a aida:Event ;\n")
                af.write(
                    '\taida:componentName "'
                    + make_xml_safe(str(data["claim_semantics"]["event"]["text"]))
                    + '"^^xsd:string ;\n'
                )
                af.write(
                    '\taida:componentIdentity "'
                    + str(data["claim_semantics"]["event"]["qnode_id"])
                    + '"^^xsd:string ;\n'
                )

                af.write(
                    '\taida:privateData "['
                    + reduce_whitespace(str(data["claim_semantics"])).replace('"', "")
                    + ']"^^xsd:string ;\n'
                )
                write_system()

                for argument, arg_data in claim_arguments.items():
                    argument_role = argument_names_to_roles[argument]

                    # EventArgument
                    af.write(argument + " a aida:EventArgument ;\n")
                    af.write(
                        '\taida:componentName "'
                        + make_xml_safe(str(arg_data["type"]["text"]))
                        + '"^^xsd:string ;\n'
                    )
                    af.write(
                        '\taida:componentType "'
                        + str(arg_data["type"]["qnode_id"])
                        + '"^^xsd:string ;\n'
                    )
                    defining_arg_qnode = str(arg_data["type"]["qnode_id"])
                    if arg_data.get("identity") is not None and arg_data["identity"]["qnode_id"] is not None:
                        af.write(
                            '\taida:componentIdentity "'
                            + str(arg_data["identity"]["qnode_id"])
                            + '"^^xsd:string ;\n'
                        )
                        defining_arg_qnode = arg_data["identity"]["qnode_id"]
                    else:
                        af.write(
                            '\taida:componentIdentity "'
                            + get_nil_id_for_entity(arg_data["type"]["text"])
                            + '"^^xsd:string ;\n'
                        )

                    af.write(
                        '\taida:privateData "['
                        + reduce_whitespace(str(arg_data)).replace('"', "")
                        + ']"^^xsd:string ;\n'
                    )
                    write_system()

                    # ArgumentStatement
                    af.write(
                        "<"
                        + cdse_system
                        + "/assertions/isi/eventarg/"
                        + source + "/"
                        + claim_semantics_id + "/"
                        + f"{data['claim_semantics']['event']['text']}.{argument_role}/"
                        + f"EventArgument-{defining_arg_qnode}"
                        + "> a rdf:Statement,\n"
                    )
                    af.write('\t\taida:ArgumentStatement ;\n')
                    af.write(f'\trdf:object {argument} ;\n')
                    af.write(f'\trdf:predicate \"{argument_role}\"^^xsd:string ;\n')
                    af.write(f'\trdf:subject {claim_semantics_event} ;\n')
                    write_system()

    if af:
        af.close()
        log.info("WROTE: %s", aif_file)


if __name__ == "__main__":
    parameters_only_entry_point(convert_json_file_to_aif)
