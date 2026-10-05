"""Phase 6: tests for the external Foursquare places-search tool layer
(itinerary.py's _parse_foursquare_response, _foursquare_search, and the
search_places tool), which had zero test coverage before this file. External
HTTP calls are stubbed via monkeypatch so these tests run offline and
deterministically.
"""

import requests

from ai_travel_planner import itinerary
from ai_travel_planner.itinerary import _foursquare_search, _parse_foursquare_response, search_places


# --- _parse_foursquare_response: structural (whole-response-invalidating) ----


def test_parse_foursquare_response_rejects_non_mapping_payload():
    assert _parse_foursquare_response(["not", "a", "dict"]) is None
    assert _parse_foursquare_response("also not a dict") is None
    assert _parse_foursquare_response(None) is None


def test_parse_foursquare_response_rejects_missing_results_key():
    assert _parse_foursquare_response({"something_else": []}) is None


def test_parse_foursquare_response_rejects_non_list_results():
    assert _parse_foursquare_response({"results": "not a list"}) is None
    assert _parse_foursquare_response({"results": {"oops": "a dict"}}) is None


def test_parse_foursquare_response_rejects_non_mapping_entry():
    # One malformed entry invalidates the WHOLE response - not just itself -
    # since it signals the API's structure changed, not one incomplete place.
    assert _parse_foursquare_response({"results": ["not a dict"]}) is None
    assert _parse_foursquare_response(
        {"results": [{"name": "Valid Place"}, "not a dict"]}
    ) is None


# --- _parse_foursquare_response: per-entry (non-invalidating) leniency -------


def test_parse_foursquare_response_skips_entries_missing_name():
    result = _parse_foursquare_response(
        {"results": [{"name": ""}, {"no_name_field": True}, {"name": "Real Place"}]}
    )
    assert result == [{"name": "Real Place", "category": None, "address": None}]


def test_parse_foursquare_response_skips_entries_with_non_string_name():
    # A truthy but non-string name (e.g. an API returning a numeric id in the
    # name field) must be skipped, not treated as a valid place name.
    result = _parse_foursquare_response(
        {"results": [{"name": 123}, {"name": "Real Place"}]}
    )
    assert result == [{"name": "Real Place", "category": None, "address": None}]


def test_parse_foursquare_response_returns_empty_list_for_all_entries_skipped():
    result = _parse_foursquare_response({"results": [{"name": ""}, {}]})
    assert result == []


def test_parse_foursquare_response_tolerates_malformed_category_metadata():
    # Malformed/missing optional metadata on an otherwise-valid entry must
    # never invalidate the whole response - only leave that field as None.
    result = _parse_foursquare_response(
        {
            "results": [
                {"name": "A", "categories": "not a list"},
                {"name": "B", "categories": {"not": "a list either"}},
                {"name": "C", "categories": []},
                {"name": "D", "categories": ["not a dict"]},
                {"name": "E", "categories": [{"no_name_key": True}]},
            ]
        }
    )
    assert result == [
        {"name": "A", "category": None, "address": None},
        {"name": "B", "category": None, "address": None},
        {"name": "C", "category": None, "address": None},
        {"name": "D", "category": None, "address": None},
        {"name": "E", "category": None, "address": None},
    ]


def test_parse_foursquare_response_tolerates_malformed_location_metadata():
    result = _parse_foursquare_response(
        {
            "results": [
                {"name": "A", "location": "not a dict"},
                {"name": "B", "location": None},
                {"name": "C", "location": {}},
            ]
        }
    )
    assert result == [
        {"name": "A", "category": None, "address": None},
        {"name": "B", "category": None, "address": None},
        {"name": "C", "category": None, "address": None},
    ]


def test_parse_foursquare_response_returns_places_for_valid_entries():
    result = _parse_foursquare_response(
        {
            "results": [
                {
                    "name": "Ramen Shop",
                    "categories": [{"name": "Ramen Restaurant"}],
                    "location": {"formatted_address": "123 Main St"},
                }
            ]
        }
    )
    assert result == [
        {"name": "Ramen Shop", "category": "Ramen Restaurant", "address": "123 Main St"}
    ]


# --- _foursquare_search: transport layer -------------------------------------


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeResponseBadJson:
    def raise_for_status(self):
        pass

    def json(self):
        raise requests.exceptions.JSONDecodeError("simulated bad JSON", "", 0)


def test_foursquare_search_returns_none_without_api_key(monkeypatch):
    monkeypatch.setattr(itinerary, "FOURSQUARE_API_KEY", None)
    assert _foursquare_search("Kyoto", "restaurant", "") is None


def test_foursquare_search_returns_none_on_request_exception(monkeypatch):
    monkeypatch.setattr(itinerary, "FOURSQUARE_API_KEY", "fake-key")

    def fake_get(*args, **kwargs):
        raise requests.Timeout("simulated timeout")

    monkeypatch.setattr(itinerary.requests, "get", fake_get)
    assert _foursquare_search("Kyoto", "restaurant", "") is None


def test_foursquare_search_returns_none_on_json_decode_error(monkeypatch):
    # Verifies response.json() being moved inside the requests.RequestException
    # try block: a decode failure must be caught there, not propagate out.
    monkeypatch.setattr(itinerary, "FOURSQUARE_API_KEY", "fake-key")
    monkeypatch.setattr(itinerary.requests, "get", lambda *a, **k: _FakeResponseBadJson())
    assert _foursquare_search("Kyoto", "restaurant", "") is None


def test_foursquare_search_returns_none_when_parser_rejects_shape(monkeypatch):
    monkeypatch.setattr(itinerary, "FOURSQUARE_API_KEY", "fake-key")
    monkeypatch.setattr(
        itinerary.requests, "get", lambda *a, **k: _FakeResponse({"results": "not a list"})
    )
    assert _foursquare_search("Kyoto", "restaurant", "") is None


def test_foursquare_search_returns_empty_list_on_no_results(monkeypatch):
    monkeypatch.setattr(itinerary, "FOURSQUARE_API_KEY", "fake-key")
    monkeypatch.setattr(itinerary.requests, "get", lambda *a, **k: _FakeResponse({"results": []}))
    assert _foursquare_search("Kyoto", "restaurant", "") == []


def test_foursquare_search_returns_parser_output_on_success(monkeypatch):
    monkeypatch.setattr(itinerary, "FOURSQUARE_API_KEY", "fake-key")
    payload = {
        "results": [
            {
                "name": "Nishiki Market",
                "categories": [{"name": "Market"}],
                "location": {"formatted_address": "Nishikikoji-dori"},
            }
        ]
    }
    monkeypatch.setattr(itinerary.requests, "get", lambda *a, **k: _FakeResponse(payload))
    assert _foursquare_search("Kyoto", "market", "") == [
        {"name": "Nishiki Market", "category": "Market", "address": "Nishikikoji-dori"}
    ]


# --- search_places tool: wraps _foursquare_search's contract into text ------


def test_search_places_tool_reports_unavailable_on_none(monkeypatch):
    monkeypatch.setattr(itinerary, "_foursquare_search", lambda city, category, query: None)
    result = search_places.invoke({"city": "Kyoto", "category": "restaurant"})
    assert result.startswith("SEARCH_UNAVAILABLE")
    assert "llm_estimate" in result


def test_search_places_tool_reports_no_results_on_empty_list(monkeypatch):
    monkeypatch.setattr(itinerary, "_foursquare_search", lambda city, category, query: [])
    result = search_places.invoke({"city": "Kyoto", "category": "restaurant"})
    assert result.startswith("SEARCH_NO_RESULTS")
    assert "llm_estimate" in result


def test_search_places_tool_formats_results_on_success(monkeypatch):
    places = [{"name": "Nishiki Market", "category": "Market", "address": "Nishikikoji-dori"}]
    monkeypatch.setattr(itinerary, "_foursquare_search", lambda city, category, query: places)
    result = search_places.invoke({"city": "Kyoto", "category": "market"})
    assert result.startswith("SEARCH_RESULTS (verified via Foursquare):")
    assert "Nishiki Market" in result
    assert "Nishikikoji-dori" in result
