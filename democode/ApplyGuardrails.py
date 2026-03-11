# This is an automatically generated code sample.
# To make this code sample work in your Oracle Cloud tenancy,
# please replace the values for any parameters whose current values do not fit
# your use case (such as resource IDs, strings containing ‘EXAMPLE’ or ‘unique_id’, and
# boolean, number, and enum parameters with values not fitting your use case).

import oci

# Create a default config using DEFAULT profile in default location
# Refer to
# https://docs.cloud.oracle.com/en-us/iaas/Content/API/Concepts/sdkconfig.htm#SDK_and_CLI_Configuration_File
# for more info
config = oci.config.from_file()


# Initialize service client with default config file
generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
    config)


# Send the request to service, some parameters are not required, see API
# doc for more info
apply_guardrails_response = generative_ai_inference_client.apply_guardrails(
    apply_guardrails_details=oci.generative_ai_inference.models.ApplyGuardrailsDetails(
        input=oci.generative_ai_inference.models.GuardrailsTextInput(
            type="TEXT",
            content="EXAMPLE-content-Value",
            language_code="EXAMPLE-languageCode-Value"),
        guardrail_configs=oci.generative_ai_inference.models.GuardrailConfigs(
            content_moderation_config=oci.generative_ai_inference.models.ContentModerationConfiguration(
                categories=["EXAMPLE--Value"]),
            personally_identifiable_information_config=oci.generative_ai_inference.models.PersonallyIdentifiableInformationConfiguration(
                types=["EXAMPLE--Value"])),
        compartment_id="ocid1.test.oc1..<unique_ID>EXAMPLE-compartmentId-Value"),
    opc_retry_token="EXAMPLE-opcRetryToken-Value",
    opc_request_id="UHSDU8YXMQ1XEONMWLSP<unique_ID>")

# Get the data from response
print(apply_guardrails_response.data)
