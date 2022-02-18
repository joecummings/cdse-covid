"""Convert the claim data from the output json into AIF files."""
from dataclasses import dataclass, field
import json
import logging
import os
from random import randint
from typing import Any, Dict, List, MutableMapping, Optional, TextIO, Tuple, Union

from aida_tools.utils import make_xml_safe, reduce_whitespace
from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

log = logging.getLogger(__name__)  # pylint:disable=invalid-name

CDSE_SYSTEM = "http://www.isi.edu/cdse"
text_to_nil_ids: Dict[str, str] = {}

AUTHOR_DATA: MutableMapping[str, Any] = {
    "claimer": {"text": "<AUTHOR>", "span": [0, 0], "confidence": 1.0},
    "claimer_identity_qnode": None,
    "claimer_type_qnode": {
        "text": "human",
        "entity": None,
        "span": [0, 0],
        "confidence": 1.0,
        "qnode_id": "Q5",
    },
}


@dataclass
class ClaimSemanticsData:
    """Base data class."""

    semantics_id: str
    name: str
    cluster_name: str
    qnode_data: Dict[str, Any]


@dataclass
class ClaimSemanticsArgData(ClaimSemanticsData):
    """A set of data for each claim semantics event for AIF conversion."""

    role: str


@dataclass
class ClaimSemanticsEventData(ClaimSemanticsData):
    """A set of data for each claim semantics event for AIF conversion."""

    arguments: List[ClaimSemanticsArgData] = field(default_factory=list)


def write_offset(aif_file: TextIO, offset: int, is_start: bool) -> None:
    """Write offset to AIF file."""
    position = "startOffset" if is_start else "endOffsetInclusive"
    aif_file.write(f'\taida:{position} "{offset}"^^xsd:int ;\n')


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
    doc_id: str,
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
        aif_file.write('\taida:sourceDocument "' + doc_id + '"^^xsd:string ;\n')
        aif_file.write('\taida:startOffset "' + str(start_offset) + '"^^xsd:int ;\n')
        write_system(aif_file)


def get_valid_claim_semantics(claim_semantics_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create a list of 'valid' claim semantics.

    Valid claim semantics events have a Qnode ID and span,
    and the Qnode is not "Unspecified."
    """
    valid_claim_semantics = []
    for claim_semantics in claim_semantics_data:
        event = claim_semantics["event"]
        event_qnode = event.get("qnode_id")
        event_span = event.get("span") and len(event["span"]) == 2
        event_text = event.get("text") != "Unspecified"
        if event and event_qnode and event_span and event_text:
            valid_claim_semantics.append(claim_semantics)
    return valid_claim_semantics


def is_valid_x_variable(x_varaible_data: Optional[Dict[str, Any]]) -> bool:
    """Returns True if the claim has a valid x-variable.

    Claim components are valid if they exist and have a span.
    """
    if not x_varaible_data:
        return False

    x_variable_span = x_varaible_data.get("span")
    if x_variable_span:
        if len(x_variable_span) != 2:
            logging.warning(
                "The span of x-variable %s is invalid (%s).", x_varaible_data, x_variable_span
            )
            return False
        return True
    return False


def get_claim_semantics_data(
    aif_file: TextIO,
    source: str,
    claim_semantics_source_data: Any,
    event_count: int,
    arg_count: int,
) -> Tuple[List[ClaimSemanticsEventData], List[str], int, int]:
    """Write basic claim semantics data to Claim and gather for later use."""
    claim_semantics_aif_data = []
    event_cluster_list = []
    arg_cluster_list = []

    for claim_event in claim_semantics_source_data:
        event_number_string = str(event_count).zfill(6)
        claim_semantics_event_id = f"EN_Event_{event_number_string}"
        claim_semantics_name = (
            "<"
            + make_xml_safe(
                CDSE_SYSTEM + f"/clusters/isi/events/{source}/{claim_semantics_event_id}"
            )
            + ">"
        )

        event_cluster_list.append(claim_semantics_name)
        event_count += 1

        claim_arguments_for_event = []
        has_claim_arguments = claim_event["args"] is not None
        if has_claim_arguments:
            argument_objects = claim_event["args"]
            for role, arg_qnodes in argument_objects.items():
                claim_argument = None
                argument_id = "EN_Entity_EDL_ENG_" + str(arg_count).zfill(7)
                if arg_qnodes.get("type"):
                    if arg_qnodes["type"]["qnode_id"] is not None and arg_qnodes["type"].get(
                        "span"
                    ):
                        claim_argument = (
                            "<"
                            + make_xml_safe(
                                CDSE_SYSTEM + "/clusters/isi/entity/" + source + "/" + argument_id
                            )
                            + ">"
                        )
                elif arg_qnodes.get("identity"):
                    if arg_qnodes["identity"]["qnode_id"] is not None and arg_qnodes["type"].get(
                        "span"
                    ):
                        claim_argument = (
                            "<"
                            + make_xml_safe(
                                CDSE_SYSTEM + "/clusters/isi/entity/" + source + "/" + argument_id
                            )
                            + ">"
                        )
                if claim_argument:
                    arg_count += 1
                    arg_cluster_list.append(claim_argument)
                    claim_arguments_for_event.append(
                        ClaimSemanticsArgData(
                            semantics_id=argument_id,
                            name="",
                            cluster_name=claim_argument,
                            qnode_data=arg_qnodes,
                            role=role,
                        )
                    )

        claim_semantics_aif_data.append(
            ClaimSemanticsEventData(
                semantics_id=claim_semantics_event_id,
                name="",
                cluster_name=claim_semantics_name,
                qnode_data=claim_event["event"],
                arguments=claim_arguments_for_event,
            )
        )

    # Write events, then arguments
    all_clusters = event_cluster_list + arg_cluster_list
    aif_file.write(f"\taida:claimSemantics {event_cluster_list.pop()}")
    for event_cluster in event_cluster_list:
        aif_file.write(f",\n\t\t{event_cluster}")
    if arg_cluster_list:
        for arg_cluster in arg_cluster_list:
            aif_file.write(f",\n\t\t{arg_cluster}")
    aif_file.write(" ;\n")

    return claim_semantics_aif_data, all_clusters, event_count, arg_count


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
    aif_file: TextIO, source: str, claim_semantics_event: ClaimSemanticsEventData, doc_id: str
) -> None:
    """Add the event of the claim and its justifications."""
    aif_file.write(claim_semantics_event.name + " a aida:Event ;\n")
    event_data = claim_semantics_event.qnode_data

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
    event_start = event_data["span"][0]
    event_end = event_data["span"][1]
    informative_justification_name = (
        "<"
        + CDSE_SYSTEM
        + f"/assertions/isi/event_informative_justification/{claim_semantics_event.semantics_id}/"
        + source
        + f"/{event_start}/{event_end}"
        + ">"
    )
    aif_file.write(f"\taida:informativeJustification {informative_justification_name} ;\n")

    # justified by
    justification_name = (
        "<"
        + CDSE_SYSTEM
        + f"/assertions/isi/eventjustification/{claim_semantics_event.semantics_id}/{source}/{event_start}/{event_end}"
        + ">"
    )
    aif_file.write(f"\taida:justifiedBy {justification_name} ;\n")

    write_private_data(aif_file, str(event_data))
    write_system(aif_file)

    # TextJustification
    aif_file.write(informative_justification_name + " a aida:TextJustification ;\n")
    aif_file.write(confidence_aif)
    write_offset(aif_file, event_end, is_start=False)
    write_private_data(aif_file, str(event_data))
    aif_file.write(f'\taida:source "{source}"^^xsd:string ;\n')
    aif_file.write(f'\taida:sourceDocument "{doc_id}"^^xsd:string ;\n')
    write_offset(aif_file, event_start, is_start=True)
    write_system(aif_file)

    # Event justification
    aif_file.write(justification_name + " a aida:TextJustification ;\n")
    aif_file.write(confidence_aif)
    write_offset(aif_file, event_end, is_start=False)
    write_private_data(aif_file, str(event_data))
    aif_file.write(f'\taida:source "{source}"^^xsd:string ;\n')
    aif_file.write(f'\taida:sourceDocument "{doc_id}"^^xsd:string ;\n')
    write_offset(aif_file, event_start, is_start=True)
    write_system(aif_file)

    # Event object
    aif_file.write(
        "<"
        + CDSE_SYSTEM
        + "/assertions/isi/eventtype/"
        + source
        + "/"
        + claim_semantics_event.semantics_id
        + "/"
        + f"{event_data['text'].replace(' ', '_')}/"
        + f"EventType-{event_data['qnode_id']}"
        + "> a rdf:Statement,\n"
    )
    aif_file.write("\t\taida:TypeStatement ;\n")
    aif_file.write(f'\trdf:object "{event_data["qnode_id"]}"^^xsd:string ;\n')
    aif_file.write("\trdf:predicate rdf:type ;\n")
    aif_file.write(f"\trdf:subject {claim_semantics_event.name} ;\n")
    write_system(aif_file)


def write_claim_semantics_argument(
    aif_file: TextIO,
    source: str,
    argument: ClaimSemanticsArgData,
    event: ClaimSemanticsEventData,
    arg_count: Union[int, str],
    doc_id: str,
) -> None:
    """Add an event argument and link it to its event."""
    # Check if there's type/identity data
    # Most args should have a type
    arg_data = argument.qnode_data
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
        aif_file, source, argument.role, argument.name, defining_arg_data, arg_count, doc_id
    )

    aif_file.write(
        "<"
        + CDSE_SYSTEM
        + "/assertions/isi/eventarg/"
        + source
        + "/"
        + event.semantics_id
        + "/"
        + f"{event.qnode_data['text'].replace(' ', '_')}.{argument.role}/"
        + f"EventArgument-{defining_arg_qnode}"
        + "> a rdf:Statement,\n"
    )
    aif_file.write("\t\taida:ArgumentStatement ;\n")
    aif_file.write(f"\trdf:object {argument.name} ;\n")
    aif_file.write(f'\trdf:predicate "{argument.role}"^^xsd:string ;\n')
    aif_file.write(f"\trdf:subject {event.name} ;\n")
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
        aida_file: TextIO,
        var_type: str,
        source: str,
        claim_id: str,
        variable_count: int,
        is_author: bool = False,
    ) -> Tuple[str, str, str, int]:
        """Get a variable's name, entity name, and justification name.

        var_type can be 'x_variable' or 'claimer'
        """
        type_for_uri = "X" if var_type == "x_variable" else var_type
        variable_name = f"ex:claim_{source}_{claim_id}_{type_for_uri}"
        variable_cluster_name = (
            (
                "<"
                + make_xml_safe(
                    CDSE_SYSTEM + "/clusters/isi/entity/" + source + "/EN_Entity_EDL_ENG_0000000"
                )
                + ">"
            )
            if is_author
            else (
                "<"
                + make_xml_safe(
                    CDSE_SYSTEM
                    + "/clusters/isi/entity/"
                    + source
                    + f"/{claim_id}/{var_type}/"
                    + str(variable_count)
                )
                + ">"
            )
        )
        variable_entity_name = (
            (
                "<"
                + make_xml_safe(CDSE_SYSTEM + f"/entities/isi/{source}/EN_Entity_EDL_ENG_0000000")
                + ">"
            )
            if is_author
            else (
                "<"
                + make_xml_safe(
                    CDSE_SYSTEM
                    + "/"
                    + source
                    + "/claim/"
                    + claim_id
                    + f"/{var_type}/entity/"
                    + str(variable_count)
                )
                + ">"
            )
        )
        aida_file.write(f"\taida:{var_types_to_aida_classes[var_type]} " + variable_name + " ;\n")
        associated_kes.append(variable_cluster_name)
        variable_count += 1
        return variable_name, variable_entity_name, variable_cluster_name, variable_count

    def write_qnode_data(
        aif_file: TextIO,
        data: Any,
        var_type: str,
        variable_name: str,
        variable_entity_name: str,
        variable_cluster_name: str,
        var_count: int,
        doc_id: str,
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

        # Cluster
        aif_file.write(variable_cluster_name + " a aida:SameAsCluster ;\n")
        aif_file.write(f"\taida:prototype {variable_entity_name} ;\n")
        write_system(aif_file)

        # Claim component
        aif_file.write(variable_name + " a aida:ClaimComponent ;\n")
        aif_file.write(
            '\taida:componentName "'
            + make_xml_safe(str(variable_entity_data["text"])).replace("\\", "_")
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
            doc_id,
        )

    event_count = 0
    entity_count = 1  # 0 is reserved for the "author"

    for data in claims_data:
        source = os.path.splitext(str(data["doc_id"]))[0]

        if source != prior_source:
            if prior_source != "":
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
            # First check if there is at least one valid claim semantics object
            claim_semantics = get_valid_claim_semantics(data["claim_semantics"])
            if not claim_semantics:
                continue

            # Each claim needs at least one x-variable
            if not is_valid_x_variable(data["x_variable_type_qnode"]):
                continue

            # Write the Claim
            claim_id = make_xml_safe(str(data["claim_id"]))
            claim_name = (
                "<" + make_xml_safe(CDSE_SYSTEM + "/" + source + "/claim_id/" + claim_id) + ">"
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
            x_variable_cluster_name = "None"

            associated_kes = []

            if has_x_variable:
                (
                    x_variable_name,
                    x_variable_entity_name,
                    x_variable_cluster_name,
                    entity_count,
                ) = get_name_data(af, "x_variable", source, claim_id, entity_count)

            af.write(
                '\taida:naturalLanguageDescription "'
                + reduce_whitespace(str(data["claim_text"])).replace('"', "")
                + '"^^xsd:string ;\n'
            )

            (
                claim_semantics_data,
                semantics_kes,
                event_count,
                entity_count,
            ) = get_claim_semantics_data(af, source, claim_semantics, event_count, entity_count)
            associated_kes.extend(semantics_kes)

            has_claimer = data["claimer_type_qnode"] is not None
            claimer_name, claimer_entity_name, claimer_cluster_name, entity_count = get_name_data(
                af, "claimer", source, claim_id, entity_count, is_author=(not has_claimer)
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
            af.write('\taida:sourceDocument "' + str(data["doc_id"]) + '"^^xsd:string ;\n')
            af.write('\taida:startOffset "' + str(start_offset) + '"^^xsd:int ;\n')
            af.write(f"\taida:system <{CDSE_SYSTEM}> . \n\n")

            if has_x_variable:
                write_qnode_data(
                    af,
                    data,
                    "x_variable",
                    x_variable_name,
                    x_variable_entity_name,
                    x_variable_cluster_name,
                    0,
                    data["doc_id"],
                )

            AUTHOR_DATA["claimer"]["doc_id"] = source

            write_qnode_data(
                af,
                data if has_claimer else AUTHOR_DATA,
                "claimer",
                claimer_name,
                claimer_entity_name,
                claimer_cluster_name,
                0,
                data["doc_id"],
            )

            if has_claim_location:
                write_claim_component(af, data, "claim_location", claim_location)

            for claim_semantics_event in claim_semantics_data:
                # SameAsCluster
                af.write(claim_semantics_event.cluster_name + " a aida:SameAsCluster ;\n")
                claim_semantics_event_name = (
                    "<"
                    + CDSE_SYSTEM
                    + f"/events/isi/{source}/{claim_semantics_event.semantics_id}"
                    + ">"
                )
                af.write("\taida:prototype " + claim_semantics_event_name + " ;\n")
                claim_semantics_event.name = claim_semantics_event_name
                write_system(af)

                # Event
                write_claim_semantics_event(af, source, claim_semantics_event, data["doc_id"])

                for argument in claim_semantics_event.arguments:
                    # Argument SameAsClusters
                    af.write(argument.cluster_name + "a aida:SameAsCluster ;\n")
                    claim_semantics_arg_name = (
                        "<" + CDSE_SYSTEM + f"/entities/isi/{source}/{argument.semantics_id}" + ">"
                    )
                    af.write("\taida:prototype " + claim_semantics_arg_name + " ;\n")
                    argument.name = claim_semantics_arg_name
                    write_system(af)

                    # Arguments
                    arg_number = claim_semantics_arg_name.split("_")[-1].strip(">")

                    write_claim_semantics_argument(
                        af, source, argument, claim_semantics_event, arg_number, data["doc_id"]
                    )

        if af:
            af.write(f"<{CDSE_SYSTEM}> a aida:System .")

    if af:
        af.close()
        log.info("WROTE: %s", aif_file)


if __name__ == "__main__":
    parameters_only_entry_point(convert_json_file_to_aif)
