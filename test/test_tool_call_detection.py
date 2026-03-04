from src.utils.json_helper import detect_all_tool_call_blocks


def test_detect_tool_call_strict_tag():
  text = """
<TOOL_CALL>
{"name":"Read","input":{"file_path":"README.txt"}}
</TOOL_CALL>
"""
  calls = detect_all_tool_call_blocks(text)
  assert len(calls) == 1
  assert calls[0]["name"] == "Read"
  assert calls[0]["input"]["file_path"] == "README.txt"


def test_detect_tool_call_tag_with_suffix_attributes():
  text = """
<TOOL_CALL  measures_json>
{"name":"TaskGet","input":{"taskId":"task_123"}}
</TOOL_CALL>
"""
  calls = detect_all_tool_call_blocks(text)
  assert len(calls) == 1
  assert calls[0]["name"] == "TaskGet"
  assert calls[0]["input"]["taskId"] == "task_123"