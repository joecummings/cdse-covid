"""Convert the claim data from the output json into AIF files."""
import json
import logging
import os

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

    prior_source = ""
    x_count = 0

    af = None
    aif_file = None
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
            af.write("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n\n")
            x_count = 0
            prior_source = source
        if af:
            claim_id = make_xml_safe(str(data["claim_id"]))
            claim_name = (
                "<"
                + make_xml_safe("http://www.isi.edu/gaia/" + source + "/claim_id/" + claim_id)
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

            has_x_variable = data["x_variable_type_qnode"] is not None
            x_variable_name = "None"
            x_variable_entity_name = "None"
            x_variable_justification_name = "None"

            if has_x_variable:
                x_variable_name = (
                    "<"
                    + make_xml_safe(
                        "http://www.isi.edu/gaia/"
                        + source
                        + "/claim/"
                        + claim_id
                        + "/x_variable/"
                        + str(x_count)
                    )
                    + ">"
                )
                x_variable_entity_name = (
                    "<"
                    + make_xml_safe(
                        "http://www.isi.edu/gaia/"
                        + source
                        + "/claim/"
                        + claim_id
                        + "/x_variable/entity/"
                        + str(x_count)
                    )
                    + ">"
                )
                if data["x_variable_type_qnode"]["span"] is not None:
                    start_offset = data["x_variable_type_qnode"]["span"][0]
                    end_offset_inclusive = data["x_variable_type_qnode"]["span"][1]
                    x_variable_justification_name = (
                        "<"
                        + make_xml_safe(
                            "http://www.isi.edu/gaia/"
                            + source
                            + "/x_variable/justification"
                            + str(x_count)
                            + "/"
                            + str(start_offset)
                            + "/"
                            + str(end_offset_inclusive)
                        )
                        + ">"
                    )
                x_count += 1
                af.write("\taida:xVariable " + x_variable_name + " ;\n")

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
            if has_claim_semantics:
                claim_semantics = (
                    "<"
                    + make_xml_safe(
                        "http://www.isi.edu/gaia/events/isi/qnode/claim-semantics-"
                        + str(data["claim_semantics"]["event"]["qnode_id"])
                    )
                    + ">"
                )
                af.write("\taida:claimSemantics " + claim_semantics + " ;\n")

            has_claimer = data["claimer_type_qnode"] is not None
            claimer_name = "None"
            if has_claimer:
                claimer_name = (
                    "<"
                    + make_xml_safe(
                        "http://www.isi.edu/gaia/"
                        + source
                        + "/claimer/"
                        + str(data["claimer_type_qnode"]["qnode_id"])
                    )
                    + ">"
                )
                af.write("\taida:claimer " + claimer_name + " ;\n")
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
                    "http://www.isi.edu/gaia/claims/justifications/"
                    + source
                    + "/"
                    + str(start_offset)
                    + "/"
                    + str(end_offset_inclusive)
                )
                + ">"
            )

            af.write("\taida:justifiedBy " + claim_justification_name + " ;\n")
            # NOTE: claimText is not in AIF spec, but included for completeness
            af.write('\taida:claimText "' + str(data["claim_text"]) + '"^^xsd:string .\n\n')

            af.write(claim_justification_name + " a aida:TextJustification ;\n")
            af.write('\taida:endOffsetInclusive "' + str(end_offset_inclusive) + '"^^xsd:int ;\n')
            af.write('\taida:source "' + source + '"^^xsd:string ;\n')
            af.write('\taida:startOffset "' + str(start_offset) + '"^^xsd:int ;\n')
            af.write("\taida:system <http://www.isi.edu> . \n\n")

            if has_x_variable:
                af.write(x_variable_name + " a aida:ClaimComponent ;\n")
                af.write(
                    '\taida:componentName "'
                    + make_xml_safe(str(data["x_variable_type_qnode"]["text"]))
                    + '"^^xsd:string ;\n'
                )
                af.write(
                    '\taida:componentType "'
                    + str(data["x_variable_type_qnode"]["qnode_id"])
                    + '"^^xsd:string ;\n'
                )
                if (
                    data["x_variable_identity_qnode"] is not None
                    and data["x_variable_identity_qnode"]["qnode_id"] is not None
                ):
                    af.write(
                        '\taida:componentIdentity "'
                        + str(data["x_variable_identity_qnode"]["qnode_id"])
                        + '"^^xsd:string ;\n'
                    )
                af.write("\taida:componentKE " + x_variable_entity_name + " ;\n")
                af.write(
                    '\taida:privateData "['
                    + reduce_whitespace(str(data["x_variable_type_qnode"]))
                    + ']"^^xsd:string .\n\n'
                )

                af.write(x_variable_entity_name + " a aida:Entity ;\n")
                af.write(
                    '\taida:text "'
                    + make_xml_safe(str(data["x_variable_type_qnode"]["text"]))
                    + '"^^xsd:string ;\n'
                )
                af.write(
                    '\taida:name "'
                    + str(data["x_variable_type_qnode"]["qnode_id"])
                    + '"^^xsd:string ;\n'
                )
                af.write("\taida:justifiedBy " + x_variable_justification_name + " ;\n")
                if data["x_variable"]["entity"] is not None:
                    af.write(
                        '\taida:privateData "['
                        + reduce_whitespace(str(data["x_variable"]["entity"]))
                        + ']"^^xsd:string ;\n'
                    )
                # TODO add x_variable_type_qnode
                af.write("\taida:system <https://www.isi.edu> . \n\n")

                if data["x_variable_type_qnode"]["span"] is not None:
                    start_offset = data["x_variable_type_qnode"]["span"][0]
                    end_offset_inclusive = data["x_variable_type_qnode"]["span"][1]
                    af.write(x_variable_justification_name + " a aida:TextJustification ;\n")
                    af.write(
                        '\taida:endOffsetInclusive "' + str(end_offset_inclusive) + '"^^xsd:int ;\n'
                    )
                    af.write('\taida:source "' + source + '"^^xsd:string ;\n')
                    af.write('\taida:startOffset "' + str(start_offset) + '"^^xsd:int ;\n')
                af.write("\taida:system <http://www.isi.edu> . \n\n")

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
                af.write(claimer_name + " a aida:ClaimComponent ;\n")
                af.write(
                    '\taida:componentName "'
                    + str(data["claimer_type_qnode"]["text"])
                    + '"^^xsd:string ;\n'
                )
                af.write(
                    '\taida:componentIdentity "'
                    + str(data["claimer_type_qnode"]["qnode_id"])
                    + '"^^xsd:string ;\n'
                )
                if (
                    data["claimer_identity_qnode"] is not None
                    and data["claimer_identity_qnode"]["qnode_id"] is not None
                ):
                    af.write(
                        '\taida:componentIdentity "'
                        + str(data["claimer_identity_qnode"]["qnode_id"])
                        + '"^^xsd:string ;\n'
                    )
                af.write(
                    '\taida:privateData "['
                    + reduce_whitespace(str(data["claimer_type_qnode"]))
                    + ']"^^xsd:string .\n\n'
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

            if has_claim_semantics:
                af.write(claim_semantics + " a aida:Event ;\n")
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
                # TODO: args|source-patient/var|:identity:qnode_id or type

                af.write(
                    '\taida:privateData "['
                    + reduce_whitespace(str(data["claim_semantics"])).replace('"', "")
                    + ']"^^xsd:string .\n\n'
                )

    if af:
        af.close()
        log.info("WROTE: %s", aif_file)


if __name__ == "__main__":
    parameters_only_entry_point(convert_json_file_to_aif)
