How to run (from repo root):

  python test/01_event_logging.py
  python test/02_count_tokens.py
  python test/03_messages_non_stream.py
  python test/04_messages_stream_sse.py
  python test/05_tools_non_stream_and_tool_result_loop.py
  python test/08_guardrails_input_block.py
  python test/09_guardrails_output_redact.py
  python test/10_guardrails_stream_downgrade.py

Config:
- Defaults to reading default_model from config.json
- Override base URL:  GATEWAY_BASE_URL=http://localhost:8000
- Override model:     GATEWAY_MODEL=your-model-alias
