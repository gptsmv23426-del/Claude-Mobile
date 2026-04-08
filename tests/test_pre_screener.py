from pre_screener import PreScreenResult, parse_pre_screen_response


def test_parse_valid_response():
    response = """MARKET: Will BTC hit 100k?
TRADEABLE: YES
REASON: Clear binary outcome, high liquidity, recent price momentum is measurable.

MARKET: Will it rain in NYC tomorrow?
TRADEABLE: NO
REASON: Weather market, unpredictable short-term, no edge possible."""
    results = parse_pre_screen_response(response, ["id1", "id2"])
    assert len(results) == 2
    assert results[0].tradeable is True
    assert results[1].tradeable is False


def test_parse_handles_partial():
    response = """MARKET: Will BTC hit 100k?
TRADEABLE: YES
REASON: Good market."""
    results = parse_pre_screen_response(response, ["id1", "id2"])
    assert len(results) == 2
    assert results[0].tradeable is True
    assert results[1].tradeable is True  # default for missing
