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
    MODEL_MAP = custom_config["model_map"]
    DEFAULT_MODEL_OCID = custom_config["default_model_ocid"]
    logger.info("Custom configuration loaded successfully from config.json")
except FileNotFoundError:
    logger.error(f"Configuration file {CONFIG_FILE} not found")
    raise
except KeyError as e:
    logger.error(f"Missing key in configuration file: {e}")
    raise
except json.JSONDecodeError:
    logger.error("Invalid JSON in configuration file")
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

async def generate_oci_stream(oci_messages, params, message_id, model_ocid):
    """
    Generate streaming response from OCI GenAI and format it as Anthropic-compatible SSE.
    """
    chat_detail = oci.generative_ai_inference.models.ChatDetails()
    chat_request = oci.generative_ai_inference.models.GenericChatRequest()
    chat_request.messages = oci_messages
    chat_request.max_tokens = params.get("max_tokens", 65535)
    chat_request.temperature = params.get("temperature", 0.7)
    chat_request.api_format = oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_GENERIC
    chat_request.is_stream = True

    chat_detail.chat_request = chat_request
    chat_detail.compartment_id = COMPARTMENT_ID
    chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=model_ocid)

    try:
        # 1. Required starting events (Claude Code needs these to initialize display)
        yield f"event: message_start\ndata: {json.dumps({
            'type': 'message_start',
            'message': {
                'id': message_id,
                'type': 'message',
                'role': 'assistant',
                'model': 'claude-3-5-sonnet-20241022',  # Fixed or dynamically from params
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
                text_chunk = data.get("message", {}).get("content", [{}])[0].get("text", "")

                if text_chunk:
                    # print("OCI chunk:", repr(text_chunk))  # Keep debug print
                    accumulated_text += text_chunk

                    # Key: include event header + correct delta type 'text_delta'
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

        # 3. Required ending sequence (otherwise Claude Code won't stop thinking/display)
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
                'input_tokens': 50  # Rough estimate, can be adjusted from params or OCI
            }
        })}\n\n"

        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        yield "data: [DONE]\n\n"  # Keep your [DONE]

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
    # Automatically handle common telemetry and token count requests (avoid 404)
    if "event_logging" in path or "count_tokens" in path:
        return {"status": "ok", "input_tokens": 10}

    # Handle messages request
    if "messages" in path:
        body = await request.json()
        model_name = body.get("model", "").lower()

        # Find matching OCID
        selected_ocid = DEFAULT_MODEL_OCID
        for key, val in MODEL_MAP.items():
            if key in model_name:
                selected_ocid = val
                break

        message_id = f"msg_oci_{uuid.uuid4().hex}"

        # Convert messages to OCI structure
        oci_msgs = []
        for m in body.get("messages", []):
            txt = m["content"] if isinstance(m["content"], str) else "".join([i.get("text", "") for i in m["content"] or []])
            oci_msg = oci.generative_ai_inference.models.Message()
            oci_msg.role = m["role"].upper()
            oci_msg.content = [oci.generative_ai_inference.models.TextContent(text=txt)]
            oci_msgs.append(oci_msg)

        if body.get("stream", False):
            return StreamingResponse(
                generate_oci_stream(oci_msgs, body, message_id, selected_ocid),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"  # Prevent proxy buffering for streaming
                }
            )
        else:
            # Non-streaming logic (omitted here, can be added later)
            return {"detail": "Non-stream not implemented yet"}

    return {"detail": "Not Found"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")