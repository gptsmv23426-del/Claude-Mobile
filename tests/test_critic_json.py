from critic import _repair_json


def test_clean_json_passes_through():
    raw = '{"concern_level": "HIGH", "counter_arguments": ["reason 1"], "overconfidence_flag": false, "rationale": "Bad trade"}'
    result = _repair_json(raw)
    assert result["concern_level"] == "HIGH"


def test_unescaped_quotes_in_rationale():
    raw = '{"concern_level": "MEDIUM", "counter_arguments": ["a"], "overconfidence_flag": false, "rationale": "The \\"evidence\\" is weak"}'
    result = _repair_json(raw)
    assert result["concern_level"] == "MEDIUM"


def test_newline_in_counter_argument():
    raw = '{"concern_level": "HIGH", "counter_arguments": ["line1\\nline2"], "overconfidence_flag": false, "rationale": "bad"}'
    result = _repair_json(raw)
    assert result["concern_level"] == "HIGH"


def test_truncated_json_extracts_concern():
    raw = '{"concern_level": "HIGH", "counter_arguments": ["reason 1", "reason 2 is very long and gets cut off by the tok'
    result = _repair_json(raw)
    assert result["concern_level"] == "HIGH"


def test_markdown_fence_stripped():
    raw = '```json\n{"concern_level": "LOW", "counter_arguments": [], "overconfidence_flag": false, "rationale": "ok"}\n```'
    result = _repair_json(raw)
    assert result["concern_level"] == "LOW"
