from src.utils.tool_normalization import normalize_tool_input, normalize_tool_name


def test_normalize_tool_name_aliases():
    assert normalize_tool_name("web_search") == "WebSearch"
    assert normalize_tool_name("multi-edit") == "MultiEdit"
    assert normalize_tool_name("Read") == "Read"


def test_normalize_tool_input_ask_user_question():
    converted = normalize_tool_input("AskUserQuestion", {"question": "hi"})
    assert converted == {"questions": ["hi"]}
