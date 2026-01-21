import os
import json
import logging
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import oci
import uvicorn

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("oci-gateway")

app = FastAPI(title="OCI GenAI Anthropic Gateway")

# ----------------------- OCI 配置 -----------------------
COMPARTMENT_ID = "ocid1.compartment.oc1..aaaaaaaakre3wvnmmhv474r2wrwlgunoeertbdi2v2tp3igwbu5sqyss3euq"
ENDPOINT = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"

# 模型映射表：将 Claude/Grok 的请求名映射到 OCI 的 OCID
MODEL_MAP = {
    "claude-3-5-sonnet": "ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceya3zoyev5tgdo3puutjfmxfnmpjutihhgqgtbyr7q6qtja",
    "claude-haiku": "ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceya3zoyev5tgdo3puutjfmxfnmpjutihhgqgtbyr7q6qtja",
    # 示例中指向同一个
    "grok": "ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceya3zoyev5tgdo3puutjfmxfnmpjutihhgqgtbyr7q6qtja"
}
DEFAULT_MODEL_OCID = "ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceya3zoyev5tgdo3puutjfmxfnmpjutihhgqgtbyr7q6qtja"

try:
    config = oci.config.from_file('~/.oci/config', "DEFAULT")
    genai_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
        config=config,
        service_endpoint=ENDPOINT,
        retry_strategy=oci.retry.NoneRetryStrategy(),
        timeout=(10, 240)
    )
    logger.info("OCI SDK 初始化成功")
except Exception as e:
    logger.error(f"配置加载失败: {e}")
    raise


# ----------------------- 核心逻辑 -----------------------

async def generate_oci_stream(oci_messages, params, message_id, model_ocid):
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
        # 1. 启动帧
        yield f"data: {json.dumps({'type': 'message_start', 'message': {'id': message_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': 'claude-3-5-sonnet', 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"
        yield f"data: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

        response = genai_client.chat(chat_detail)

        for event in response.data.events():
            if not event.data: continue
            try:
                data = json.loads(event.data)
                # 针对你的日志路径提取
                text = data.get("message", {}).get("content", [{}])[0].get("text", "")
                if text:
                    yield f"data: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text', 'text': text}})}\n\n"
            except:
                continue

        # 2. 结束帧
        yield f"data: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
        yield f"data: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'output_tokens': 0}})}\n\n"
        yield f"data: {json.dumps({'type': 'message_stop'})}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.exception("流式输出异常")
        yield f"data: {json.dumps({'type': 'error', 'error': {'message': str(e)}})}\n\n"


# ----------------------- 路由 -----------------------

@app.post("/{path:path}")
async def catch_all(path: str, request: Request):
    # 自动处理通用的 telemetry 和 token 计数请求
    if "event_logging" in path or "count_tokens" in path:
        return {"status": "ok", "input_tokens": 10}

    # 处理消息请求
    if "messages" in path:
        body = await request.json()
        model_name = body.get("model", "").lower()

        # 查找匹配的 OCID
        selected_ocid = DEFAULT_MODEL_OCID
        for key, val in MODEL_MAP.items():
            if key in model_name:
                selected_ocid = val
                break

        message_id = f"msg_oci_{uuid.uuid4().hex}"

        # 转换消息格式
        oci_msgs = []
        for m in body.get("messages", []):
            txt = m["content"] if isinstance(m["content"], str) else "".join([i.get("text", "") for i in m["content"]])
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
                    "X-Accel-Buffering": "no"  # 禁用 Nginx 等代理的缓冲
                }
            )
        else:
            # 非流式逻辑省略，结构同前...
            pass

    return {"detail": "Not Found"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")