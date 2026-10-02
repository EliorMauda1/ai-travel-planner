"""Phase 5: tests for ui_state.py, the Streamlit-agnostic orchestration glue
between the UI and the existing Phase 1-4 backend.

External calls are stubbed via monkeypatch so these tests run offline and
deterministically, without hitting OpenAI or Foursquare - same style as
tests/test_editing.py.
"""

import logging

from ai_travel_planner import intake, itinerary, ui_state
from ai_travel_planner.intake import TripRequest
from ai_travel_planner.itinerary import (
    Activity,
    DailyItinerary,
    DayPlan,
    MacroPlan,
    MacroStop,
)


def _activity(name: str, cost: float = 10.0) -> Activity:
    return Activity(
        name=name, description="d", category="sightseeing", estimated_cost_usd=cost, source="llm_estimate"
    )


def make_macro_plan() -> MacroPlan:
    return MacroPlan(
        stops=[
            MacroStop(city="Tokyo", country="Japan", days=3, rationale="r"),
            MacroStop(city="Kyoto", country="Japan", days=2, rationale="r"),
        ]
    )


def make_daily_itinerary(macro_plan: MacroPlan) -> DailyItinerary:
    days = []
    day_number = 1
    for stop in macro_plan.stops:
        for _ in range(stop.days):
            days.append(
                DayPlan(day_number=day_number, city=stop.city, activities=[_activity(f"Day {day_number}")])
            )
            day_number += 1
    return DailyItinerary(days=days, estimated_total_cost_usd=50.0)


# --- intake_turn ----------------------------------------------------------


def test_intake_turn_returns_followup_when_incomplete(monkeypatch):
    partial = TripRequest(destination="Japan", duration_days=7, traveler_count=2)

    class FakeExtractionChain:
        def invoke(self, args):
            return partial

    class FakeFollowupChain:
        def invoke(self, args):
            assert "total budget" in args["missing_fields"]

            class Msg:
                content = "What's your budget for this trip?"

            return Msg()

    monkeypatch.setattr(intake, "extraction_chain", FakeExtractionChain())
    monkeypatch.setattr(intake, "followup_chain", FakeFollowupChain())

    result, message, is_error = ui_state.intake_turn(TripRequest(), "Japan, 7 days, 2 of us")

    assert result == partial
    assert message == "What's your budget for this trip?"
    assert is_error is False


def test_intake_turn_preserves_fields_across_turns(monkeypatch):
    turn1_result = TripRequest(destination="Japan", duration_days=7, traveler_count=2, budget_usd=None)

    class FakeExtractionChainTurn1:
        def invoke(self, args):
            return turn1_result

    monkeypatch.setattr(intake, "extraction_chain", FakeExtractionChainTurn1())

    class FakeFollowupChain:
        def invoke(self, args):
            class Msg:
                content = "What's your budget?"

            return Msg()

    monkeypatch.setattr(intake, "followup_chain", FakeFollowupChain())

    result1, message1, is_error1 = ui_state.intake_turn(TripRequest(), "Japan, 7 days, 2 of us")

    assert result1 == turn1_result
    assert message1 is not None
    assert is_error1 is False

    # Turn 2: only the budget answer. The known_state passed to the chain
    # must be turn 1's result - proving fields aren't dropped between turns.
    captured_known_state = {}
    turn2_result = TripRequest(
        destination="Japan", duration_days=7, traveler_count=2, budget_usd=2000.0
    )

    class FakeExtractionChainTurn2:
        def invoke(self, args):
            captured_known_state.update(args)
            return turn2_result

    monkeypatch.setattr(intake, "extraction_chain", FakeExtractionChainTurn2())

    result2, message2, is_error2 = ui_state.intake_turn(result1, "$2000")

    assert captured_known_state["known_state"] == turn1_result.model_dump_json()
    assert result2.destination == "Japan"
    assert result2.duration_days == 7
    assert result2.traveler_count == 2
    assert result2.budget_usd == 2000.0
    assert message2 is None
    assert is_error2 is False


def test_intake_turn_handles_exception(monkeypatch, caplog):
    trip_request = TripRequest(destination="Japan")

    class FailingExtractionChain:
        def invoke(self, args):
            raise RuntimeError("network error")

    monkeypatch.setattr(intake, "extraction_chain", FailingExtractionChain())

    with caplog.at_level(logging.ERROR, logger="ai_travel_planner.ui_state"):
        result, message, is_error = ui_state.intake_turn(trip_request, "Japan")

    assert result is trip_request  # unchanged, not lost
    assert is_error is True
    assert message is not None
    assert "network error" not in message  # safe message, not the raw exception
    assert "intake_turn failed" in caplog.text  # real exception logged for diagnostics


# --- build_initial_edit_state ----------------------------------------------


def test_build_initial_edit_state_shape():
    trip_request = TripRequest(destination="Japan", duration_days=5, traveler_count=1)
    macro_plan = make_macro_plan()
    daily_itinerary = make_daily_itinerary(macro_plan)

    state = ui_state.build_initial_edit_state(trip_request, macro_plan, daily_itinerary, "less museums")

    assert set(state.keys()) == {
        "trip_request",
        "macro_plan",
        "daily_itinerary",
        "edit_request",
        "edit_scope",
        "target_stops",
        "classification_note",
        "excluded_categories",
        "edit_retry_count",
        "validation_feedback",
        "hard_constraint_violation",
        "candidate_macro_plan",
        "candidate_daily_itinerary",
        "response_message",
    }
    assert state["trip_request"] is trip_request
    assert state["macro_plan"] is macro_plan
    assert state["daily_itinerary"] is daily_itinerary
    assert state["edit_request"] == "less museums"
    assert state["edit_scope"] is None
    assert state["target_stops"] is None
    assert state["classification_note"] is None
    assert state["excluded_categories"] is None
    assert state["edit_retry_count"] == 0
    assert state["validation_feedback"] is None
    assert state["hard_constraint_violation"] is None
    assert state["candidate_macro_plan"] is None
    assert state["candidate_daily_itinerary"] is None
    assert state["response_message"] is None


# --- generate_initial_plan ---------------------------------------------------


def test_generate_initial_plan_happy_path(monkeypatch):
    macro_plan = make_macro_plan()
    daily_itinerary = make_daily_itinerary(macro_plan)
    trip_request = TripRequest(destination="Japan", duration_days=5, traveler_count=1)

    def fake_plan_trip(tr):
        assert tr is trip_request
        return {"macro_plan": macro_plan, "daily_itinerary": daily_itinerary}

    monkeypatch.setattr(itinerary, "plan_trip", fake_plan_trip)

    result_macro, result_daily, message = ui_state.generate_initial_plan(trip_request)

    assert result_macro is macro_plan
    assert result_daily is daily_itinerary
    assert message is None


def test_generate_initial_plan_handles_exception(monkeypatch, caplog):
    trip_request = TripRequest(destination="Japan", duration_days=5, traveler_count=1)

    def failing_plan_trip(tr):
        raise RuntimeError("OpenAI API error")

    monkeypatch.setattr(itinerary, "plan_trip", failing_plan_trip)

    with caplog.at_level(logging.ERROR, logger="ai_travel_planner.ui_state"):
        result_macro, result_daily, message = ui_state.generate_initial_plan(trip_request)

    assert result_macro is None
    assert result_daily is None
    assert message is not None
    assert "OpenAI API error" not in message
    assert "generate_initial_plan failed" in caplog.text


# --- run_edit_turn ------------------------------------------------------------


def test_run_edit_turn_happy_path(monkeypatch):
    macro_plan = make_macro_plan()
    daily_itinerary = make_daily_itinerary(macro_plan)
    trip_request = TripRequest(destination="Japan", duration_days=5, traveler_count=1)

    candidate_macro = make_macro_plan()
    candidate_daily = make_daily_itinerary(candidate_macro)

    class FakeEditGraph:
        def invoke(self, state):
            assert state["edit_request"] == "less museums in Kyoto"
            return {
                "edit_scope": "stop",
                "candidate_macro_plan": candidate_macro,
                "candidate_daily_itinerary": candidate_daily,
            }

    monkeypatch.setattr(itinerary, "edit_graph", FakeEditGraph())

    new_macro, new_daily, message = ui_state.run_edit_turn(
        trip_request, macro_plan, daily_itinerary, "less museums in Kyoto"
    )

    # Delegated to the real _apply_committed_edit, which commits the candidate.
    assert new_macro is candidate_macro
    assert new_daily is candidate_daily
    assert message is None


def test_run_edit_turn_unsupported_leaves_baseline_unchanged(monkeypatch):
    macro_plan = make_macro_plan()
    daily_itinerary = make_daily_itinerary(macro_plan)
    trip_request = TripRequest(destination="Japan", duration_days=5, traveler_count=1)

    class FakeEditGraph:
        def invoke(self, state):
            return {
                "edit_scope": "unsupported",
                "response_message": "I can't apply that as a plan edit.",
            }

    monkeypatch.setattr(itinerary, "edit_graph", FakeEditGraph())

    new_macro, new_daily, message = ui_state.run_edit_turn(
        trip_request, macro_plan, daily_itinerary, "what's the JR pass?"
    )

    assert new_macro is macro_plan
    assert new_daily is daily_itinerary
    assert message == "I can't apply that as a plan edit."


def test_run_edit_turn_handles_exception(monkeypatch, caplog):
    macro_plan = make_macro_plan()
    daily_itinerary = make_daily_itinerary(macro_plan)
    trip_request = TripRequest(destination="Japan", duration_days=5, traveler_count=1)

    class FailingEditGraph:
        def invoke(self, state):
            raise RuntimeError("timeout")

    monkeypatch.setattr(itinerary, "edit_graph", FailingEditGraph())

    with caplog.at_level(logging.ERROR, logger="ai_travel_planner.ui_state"):
        new_macro, new_daily, message = ui_state.run_edit_turn(
            trip_request, macro_plan, daily_itinerary, "less museums"
        )

    # Baseline preserved (identity), not corrupted or partially updated.
    assert new_macro is macro_plan
    assert new_daily is daily_itinerary
    assert message is not None
    assert "timeout" not in message
    assert "run_edit_turn failed" in caplog.text
