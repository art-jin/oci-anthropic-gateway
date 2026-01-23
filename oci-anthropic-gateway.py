import os
import json
import logging
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import oci
import uvicorn

# --- Logging configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("oci-gateway")

app = FastAPI(title="OCI GenAI Anthropic Gateway")

# ----------------------- Load Custom Configuration from File -----------------------
CONFIG_FILE = "config.json"  # Configuration file name (in the same directory)

try:
    with open(CONFIG_FILE, "r") as f:
        custom_config = json.load(f)

    COMPARTMENT_ID = custom_config["compartment_id"]
    ENDPOINT = custom_config["endpoint"]
    MODEL_ALIASES = custom_config.get("model_aliases", {})
    MODEL_DEFINITIONS = custom_config.get("model_definitions", {})
    DEFAULT_MODEL_NAME = custom_config.get("default_model")

    # Get the complete configuration object for the default model
    DEFAULT_MODEL_CONF = MODEL_DEFINITIONS.get(DEFAULT_MODEL_NAME)

    if not DEFAULT_MODEL_CONF:
        logger.error(f"Default model '{DEFAULT_MODEL_NAME}' not found in definitions")
        raise KeyError("Invalid default_model")

    logger.info(f"Config loaded. Default model: {DEFAULT_MODEL_NAME}")
except Exception as e:
    logger.error(f"Configuration initialization failed: {e}")
    raise

# Load OCI SDK configuration from default file
try:
    sdk_config = oci.config.from_file('~/.oci/config', "DEFAULT")
    genai_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
        config=sdk_config,
        service_endpoint=ENDPOINT,
        retry_strategy=oci.retry.NoneRetryStrategy(),
        timeout=(10, 240)
    )
    logger.info("OCI SDK initialized successfully")
except Exception as e:
    logger.error(f"SDK configuration loading failed: {e}")
    raise


# ----------------------- Core streaming generation function -----------------------

async def generate_oci_stream(oci_messages, params, message_id, model_conf, requested_model):
    """
    Generate streaming response from OCI GenAI and format it as Anthropic-compatible SSE.
    """
    chat_detail = oci.generative_ai_inference.models.ChatDetails()
    chat_request = oci.generative_ai_inference.models.GenericChatRequest()
    chat_request.messages = oci_messages

    # --- Dynamic Parameter Adaptation ---
    # 1. Handle Max Tokens (determine which key to use based on model definition)
    token_limit = params.get("max_tokens", 131070)  # 65535 * 2
    tokens_key = model_conf.get("max_tokens_key", "max_tokens")
    setattr(chat_request, tokens_key, token_limit)

    # 2. Handle Temperature (prioritize hardcoded value in model definition, otherwise use request value)
    # Addresses previous 400 errors: if model definition specifies 1.0, force 1.0
    chat_request.temperature = model_conf.get("temperature", params.get("temperature", 0.7))

    chat_request.api_format = oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_GENERIC
    chat_request.is_stream = True

    chat_detail.chat_request = chat_request
    chat_detail.compartment_id = COMPARTMENT_ID
    # Use the OCID from the configuration object
    chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=model_conf["ocid"])

    try:
        # 1. Required starting events (Claude Code needs these to initialize display)
        yield f"event: message_start\ndata: {json.dumps({
            'type': 'message_start',
            'message': {
                'id': message_id,
                'type': 'message',
                'role': 'assistant',
                'model': requested_model,
                'content': [],
                'stop_reason': None,
                'stop_sequence': None,
                'usage': {'input_tokens': 10, 'output_tokens': 0}  # Rough estimate for input
            }
        })}\n\n"

        yield f"event: content_block_start\ndata: {json.dumps({
            'type': 'content_block_start',
            'index': 0,
            'content_block': {'type': 'text', 'text': ''}
        })}\n\n"

        # Call OCI
        response = genai_client.chat(chat_detail)

        accumulated_text = ""  # Used for usage estimation and debugging

        for event in response.data.events():
            if not event.data:
                continue
            try:
                data = json.loads(event.data)
                # Adapt to varying response structures across different models
                choices = data.get("choices", [])
                if choices:
                    text_chunk = choices[0].get("message", {}).get("content", "")
                else:
                    text_chunk = data.get("message", {}).get("content", [{}])[0].get("text", "")

                if text_chunk:
                    accumulated_text += text_chunk

                    # Include event header + correct delta type 'text_delta'
                    yield f"event: content_block_delta\ndata: {json.dumps({
                        'type': 'content_block_delta',
                        'index': 0,
                        'delta': {
                            'type': 'text_delta',
                            'text': text_chunk
                        }
                    })}\n\n"

            except json.JSONDecodeError:
                logger.warning("Invalid JSON chunk, skipped")
                continue
            except Exception as e:
                logger.warning(f"Chunk processing exception: {e}")
                continue

        # 3. Required ending sequence (prevents client from "thinking" indefinitely)
        yield f"event: content_block_stop\ndata: {json.dumps({
            'type': 'content_block_stop',
            'index': 0
        })}\n\n"

        estimated_output_tokens = max(1, len(accumulated_text) // 4)  # Rough estimation
        yield f"event: message_delta\ndata: {json.dumps({
            'type': 'message_delta',
            'delta': {
                'stop_reason': 'end_turn',
                'stop_sequence': None
            },
            'usage': {
                'output_tokens': estimated_output_tokens,
                'input_tokens': 50  # Rough estimate
            }
        })}\n\n"

        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.exception("Streaming output exception")
        yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'message': str(e)}})}\n\n"


# ----------------------- Routes -----------------------

@app.post("/{path:path}")
async def catch_all(path: str, request: Request):
    """
    Catch-all route to handle all incoming paths.
    Handles telemetry, token count, and messages requests.
    """
    # Automatically handle common telemetry and token count requests to avoid 404
    if "event_logging" in path or "count_tokens" in path:
        return {"status": "ok", "input_tokens": 10}

    # Handle messages request
    if "messages" in path:
        body = await request.json()
        req_model = body.get("model", "").lower()

        # Lookup logic
        target_alias = None
        for k, v in MODEL_ALIASES.items():
            if k in req_model:
                target_alias = v
                break

        if not target_alias and req_model in MODEL_DEFINITIONS:
            target_alias = req_model

        # Retrieve the complete dictionary object, e.g., {"ocid": "...", "temperature": 1.0}
        selected_model_conf = MODEL_DEFINITIONS.get(target_alias)

        if not selected_model_conf:
            selected_model_conf = DEFAULT_MODEL_CONF
            logger.info(f"Using default model config for: {req_model}")
        else:
            logger.info(f"Mapped '{req_model}' -> '{target_alias}' config")

        # Convert messages to OCI format
        oci_msgs = []
        for m in body.get("messages", []):
            txt = m["content"] if isinstance(m["content"], str) else "".join([i.get("text", "") for i in m["content"]])
            msg = oci.generative_ai_inference.models.Message(role=m["role"].upper(), content=[
                oci.generative_ai_inference.models.TextContent(text=txt)])
            oci_msgs.append(msg)

        message_id = f"msg_oci_{uuid.uuid4().hex}"

        if body.get("stream", False):
            return StreamingResponse(
                generate_oci_stream(oci_msgs, body, message_id, selected_model_conf, req_model),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"  # Prevent proxy buffering for streaming
                }
            )
        else:
            # Non-streaming logic (to be implemented)
            return {"detail": "Non-stream not implemented yet"}

    return {"detail": "Not Found"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")