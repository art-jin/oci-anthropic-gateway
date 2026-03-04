from src.utils.json_helper import detect_all_tool_call_blocks


def _assert_eq(a, b, msg: str) -> None:
  if a != b:
      raise AssertionError(f"{msg}: {a!r} != {b!r}")


def main() -> None:
  # Case 1: strict tag
  text1 = """

"""
  calls1 = detect_all_tool_call_blocks(text1)
  _assert_eq(len(calls1), 1, "strict tag tool call count")
  _assert_eq(calls1[0]["name"], "Read", "strict tag tool name")
  _assert_eq(calls1[0]["input"]["file_path"], "README.md", "strict tag input.file_path")

  # Case 2: tag with suffix/attributes (what broke streaming before)
  text2 = """
<TOOL_CALL  measures_json>
{"name":"TaskGet","input":{"taskId":"task_123"}}
</TOOL_CALL>
"""
  calls2 = detect_all_tool_call_blocks(text2)
  _assert_eq(len(calls2), 1, "suffix tag tool call count")
  _assert_eq(calls2[0]["name"], "TaskGet", "suffix tag tool name")
  _assert_eq(calls2[0]["input"]["taskId"], "task_123", "suffix tag input.taskId")

  print("OK: 06_tool_call_detection_suffix")


if __name__ == "__main__":
  main()
