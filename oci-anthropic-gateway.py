# oci-anthropic-gateway.py
# Uses OCI Python SDK default config (~/.oci/config) + Signer for request signing.
# Supports Claude Code calls to OCI GenAI OpenAI-compatible endpoints via Anthropic API style.

import os
import json
import logging
from typing import Dict, Any, AsyncGenerator
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
import httpx
import oci
from oci.signer import Signer

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("oci-gateway")

app = FastAPI(title="OCI GenAI ← Anthropic API Gateway (using OCI SDK default config)")

# ----------------------- Configuration -----------------------
# Load from DEFAULT profile in ~/.oci/config
try:
    config = oci.config.from_file(file_location=os.path.expanduser("~/.oci/config"), profile_name="DEFAULT")
except Exception as e:
    raise RuntimeError(f"Failed to load OCI config from ~/.oci/config: {e}")

# Region is automatically retrieved from config
region = config.get("region", "us-chicago-1")
oci_base_url = f"https://inference.generativeai.{region}.oci.oraclecloud.com"

# OCI GenAI OpenAI-compatible endpoint path
OCI_CHAT_COMPLETIONS_PATH = "/20231130/actions/v1/chat/completions"

# COMPARTMENT_ID logic: retrieved from environment or config file
COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_ID") or config.get("compartment_id")
if not COMPARTMENT_ID:
    raise ValueError("Please set OCI_COMPARTMENT_ID env var or add 'compartment_id' to your ~/.oci/config")

# Create Signer for API key authentication
signer: Signer = oci.signer.Signer(
    tenancy=config["tenancy"],
    user=config["user"],
    fingerprint=config["fingerprint"],
    private_key_file_location=config["key_file"],
    pass_phrase=config.get("pass_phrase")  # Required only if private key has a passphrase
)


# ----------------------- Utility Functions -----------------------
def get_signed_headers(method: str, path: str, body: dict = None) -> Dict:
    """Generates signed headers using OCI Signer"""
    full_url = f"{oci_base_url}{path}"
    request = httpx.Request(method, full_url, json=body)
    # Signer modifies the request headers in-place
    signer.sign_request(request)
    return dict(request.headers)


async def oci_chat_completions(messages: list, model: str, stream: bool = False, **kwargs) -> AsyncGenerator:
    url = f"{oci_base_url}{OCI_CHAT_COMPLETIONS_PATH}"
    logger.info(f"Requesting OCI: {url} | model={model} | stream={stream}")

    payload = {
        "model": model,
        "messages": messages,
        "compartmentId": COMPARTMENT_ID,
        "stream": stream,
        **kwargs
    }

    try:
        headers = get_signed_headers("POST", OCI_CHAT_COMPLETIONS_PATH, payload)
        headers["Content-Type"] = "application/json"
        logger.debug(f"Generated headers: {headers}")
    except Exception as e:
        logger.error(f"Signing failed: {e}", exc_info=True)
        raise

    timeout = httpx.Timeout(60.0, connect=10.0, read=180.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            if stream:
                logger.info("Starting streaming request")
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    logger.info(f"Stream response status: {response.status_code}")
                    response.raise_for_status()

                    line_count = 0
                    async for line in response.aiter_lines():
                        line_count += 1
                        if line_count % 20 == 0:
                            logger.debug(f"Received {line_count} lines of stream data")

                        if line.startswith("data: "):
                            data = line[6:].strip()
                            if data == "[DONE]":
                                logger.info("Received [DONE] signal")
                                break
                            try:
                                chunk = json.loads(data)
                                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                            except json.JSONDecodeError as e:
                                logger.warning(f"Invalid chunk: {data[:100]} | error: {e}")
                                continue
                    logger.info("Streaming finished")
                    yield "data: [DONE]\n\n"

            else:
                logger.info("Starting non-stream request")
                resp = await client.post(url, headers=headers, json=payload)
                logger.info(f"Non-stream response status: {resp.status_code}")
                logger.debug(f"Response preview: {resp.text[:400]}")

                resp.raise_for_status()

                try:
                    data = resp.json()
                    logger.info("Successfully parsed JSON response")
                    yield data
                except json.JSONDecodeError as e:
                    logger.error(f"Response is not valid JSON: {resp.text[:500]}")
                    raise ValueError(f"OCI returned non-JSON content: {resp.text[:200]}") from e

        except httpx.TimeoutException:
            logger.error("OCI request timed out")
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"OCI HTTP Error {e.response.status_code}: {e.response.text}")
            raise
        except Exception:
            logger.exception("Unknown exception in oci_chat_completions")
            raise


# ----------------------- Format Conversion -----------------------
def convert_anthropic_to_openai_messages(anthropic_messages: list) -> list:
    openai_msgs = []
    system_content = None

    for msg in anthropic_messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            system_content = content
            continue

        if isinstance(content, str):
            openai_msgs.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Simplification: Handle text parts only (expandable for image/tool_use)
            text_parts = [part["text"] for part in content if part.get("type") == "text"]
            openai_msgs.append({"role": role, "content": "\n".join(text_parts)})

    if system_content:
        openai_msgs.insert(0, {"role": "system", "content": system_content})

    return openai_msgs


# ----------------------- Endpoints -----------------------

@app.post("/v1/api/event_logging/batch")
async def ignore_telemetry():
    """Endpoint to suppress Claude Code telemetry 404 errors"""
    return {"status": "ok"}

@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    body = await request.json()

    model = body.get("model")
    if not model:
        raise HTTPException(400, "model is required")

    # Model Mapping (Anthropic Name → OCI Model Name)
    # Example: claude-3-5-sonnet-20241022 → meta.llama-3.1-70b-instruct
    OCI_MODEL_MAP = {
        "claude-3-5-sonnet-20241022": "meta.llama-3.1-70b-instruct",
        "claude-3-opus-20240229": "cohere.command-r-plus",
    }
    oci_model = OCI_MODEL_MAP.get(model, model)

    anthropic_messages = body.get("messages", [])
    openai_messages = convert_anthropic_to_openai_messages(anthropic_messages)

    stream = body.get("stream", False)
    max_tokens = body.get("max_tokens", 4096)
    temperature = body.get("temperature", 0.7)
    top_p = body.get("top_p")

    kwargs = {
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if top_p is not None:
        kwargs["top_p"] = top_p

    if stream:
        return StreamingResponse(
            oci_chat_completions(openai_messages, oci_model, stream=True, **kwargs),
            media_type="text/event-stream"
        )

    else:
        # Retrieve the first (and only) item from the async generator
        generator = oci_chat_completions(openai_messages, oci_model, stream=False, **kwargs)
        result = await generator.__anext__()

        # Transform to Anthropic-style response (Simplified)
        choice = result["choices"][0]
        content = choice["message"]["content"]

        return {
            "id": result.get("id", "msg_oci_proxy"),
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": content}],
            "stop_reason": choice.get("finish_reason", "end_turn"),
            "usage": result.get("usage", {}),
        }


if __name__ == "__main__":
    import uvicorn
    # Start server
    uvicorn.run(app, host="0.0.0.0", port=8000)