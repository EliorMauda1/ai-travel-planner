"""Phase 6: tests for the initial macro-generation graph (generate_macro_plan,
validate_macro_plan, route_after_validation, itinerary_graph, plan_trip).

This is step 1 of the MVP's core loop and, before this file, had zero test
coverage - the exact class of invariant (stop days summing to duration_days)
that Fix 2 hardened on the daily-itinerary/edit side had no equivalent
protection here. External chains/tools are stubbed via monkeypatch so these
tests run offline and deterministically, matching tests/test_editing.py.
"""

from ai_travel_planner import itinerary
from ai_travel_planner.intake import TripRequest
from ai_travel_planner.itinerary import (
    MacroPlan,
    MacroStop,
    StopDayContent,
    StopItinerary,
    generate_macro_plan,
    plan_trip,
    route_after_validation,
    validate_macro_plan,
)


def make_trip_request(duration_days: int = 7) -> TripRequest:
    return TripRequest(
        destination="Japan",
        duration_days=duration_days,
        budget_usd=2000,
        traveler_count=2,
        interests=["food", "culture"],
        pace="moderate",
    )


def make_macro_plan(total_days: int = 7) -> MacroPlan:
    return MacroPlan(
        stops=[
            MacroStop(city="Tokyo", country="Japan", days=total_days - 2, rationale="r"),
            MacroStop(city="Kyoto", country="Japan", days=2, rationale="r"),
        ]
    )


# --- generate_macro_plan -----------------------------------------------------


def test_generate_macro_plan_invokes_chain_with_trip_request(monkeypatch):
    trip_request = make_trip_request()
    macro_plan = make_macro_plan()
    captured = {}

    class FakeMacroChain:
        def invoke(self, args):
            captured.update(args)
            return macro_plan

    monkeypatch.setattr(itinerary, "macro_chain", FakeMacroChain())

    state = {
        "trip_request": trip_request,
        "macro_plan": None,
        "macro_retry_count": 0,
        "validation_feedback": None,
        "daily_itinerary": None,
    }
    result = generate_macro_plan(state)

    assert result == {"macro_plan": macro_plan}
    assert captured["trip_request"] == trip_request.model_dump_json()
    assert captured["feedback"] == "This is the first attempt."
    assert captured["current_plan"] == ""


def test_generate_macro_plan_passes_existing_validation_feedback(monkeypatch):
    captured = {}

    class FakeMacroChain:
        def invoke(self, args):
            captured.update(args)
            return make_macro_plan()

    monkeypatch.setattr(itinerary, "macro_chain", FakeMacroChain())

    state = {
        "trip_request": make_trip_request(),
        "macro_plan": None,
        "macro_retry_count": 1,
        "validation_feedback": "Previous attempt summed to 8, expected 7.",
        "daily_itinerary": None,
    }
    generate_macro_plan(state)

    assert captured["feedback"] == "Previous attempt summed to 8, expected 7."


# --- validate_macro_plan ------------------------------------------------------


def test_validate_macro_plan_passes_when_sum_matches():
    trip_request = make_trip_request(duration_days=7)
    macro_plan = make_macro_plan(total_days=7)

    state = {
        "trip_request": trip_request,
        "macro_plan": macro_plan,
        "macro_retry_count": 0,
        "validation_feedback": None,
        "daily_itinerary": None,
    }
    result = validate_macro_plan(state)

    assert result == {"validation_feedback": None}


def test_validate_macro_plan_fails_and_increments_retry_when_sum_mismatches():
    trip_request = make_trip_request(duration_days=7)
    macro_plan = make_macro_plan(total_days=8)  # sums to 8, trip is 7

    state = {
        "trip_request": trip_request,
        "macro_plan": macro_plan,
        "macro_retry_count": 0,
        "validation_feedback": None,
        "daily_itinerary": None,
    }
    result = validate_macro_plan(state)

    assert result["validation_feedback"] is not None
    assert "8" in result["validation_feedback"]
    assert "7" in result["validation_feedback"]
    assert result["macro_retry_count"] == 1


# --- route_after_validation ----------------------------------------------------


def test_route_after_validation_valid_when_no_feedback():
    state = {"validation_feedback": None, "macro_retry_count": 0}
    assert route_after_validation(state) == "valid"


def test_route_after_validation_retries_under_cap():
    state = {"validation_feedback": "fix it", "macro_retry_count": 0}
    assert route_after_validation(state) == "retry"


def test_route_after_validation_stops_retrying_at_cap():
    state = {"validation_feedback": "fix it", "macro_retry_count": itinerary.MAX_MACRO_RETRIES}
    assert route_after_validation(state) == "valid"


# --- itinerary_graph topology ---------------------------------------------------


def test_itinerary_graph_topology_unchanged():
    nodes = set(itinerary.itinerary_graph.get_graph().nodes.keys())
    assert nodes == {
        "__start__",
        "generate_macro_plan",
        "validate_macro_plan",
        "generate_daily_itinerary",
        "__end__",
    }


# --- plan_trip end-to-end happy path --------------------------------------------


def test_plan_trip_end_to_end_happy_path(monkeypatch):
    trip_request = make_trip_request(duration_days=5)
    macro_plan = MacroPlan(
        stops=[
            MacroStop(city="Tokyo", country="Japan", days=3, rationale="r"),
            MacroStop(city="Kyoto", country="Japan", days=2, rationale="r"),
        ]
    )

    class FakeMacroChain:
        def invoke(self, args):
            return macro_plan

    monkeypatch.setattr(itinerary, "macro_chain", FakeMacroChain())

    def fake_plan_stop(trip_request, stop, extra_instruction=None):
        return StopItinerary(
            days=[
                StopDayContent(
                    activities=[
                        itinerary.Activity(
                            name=f"{stop.city} activity",
                            description="d",
                            category="sightseeing",
                            estimated_cost_usd=10.0,
                            source="llm_estimate",
                        )
                    ]
                )
                for _ in range(stop.days)
            ]
        )

    monkeypatch.setattr(itinerary, "_plan_stop", fake_plan_stop)

    result = plan_trip(trip_request)

    assert result["macro_plan"] == macro_plan
    daily_itinerary = result["daily_itinerary"]
    assert len(daily_itinerary.days) == 5
    assert [d.day_number for d in daily_itinerary.days] == [1, 2, 3, 4, 5]
    assert [d.city for d in daily_itinerary.days] == ["Tokyo"] * 3 + ["Kyoto"] * 2
    assert daily_itinerary.estimated_total_cost_usd == 50.0
