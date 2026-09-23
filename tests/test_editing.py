"""Phase 4: tests for the conversational plan-editing graph.

External calls (edit_classify_chain, macro_chain, _plan_stop) are stubbed via
monkeypatch so these tests run offline and deterministically, without hitting
OpenAI or Foursquare.
"""

import pytest

from ai_travel_planner import itinerary
from ai_travel_planner.intake import TripRequest
from ai_travel_planner.itinerary import (
    Activity,
    DailyItinerary,
    DayPlan,
    EditClassification,
    MacroPlan,
    MacroStop,
    StopDayContent,
    StopItinerary,
    _apply_committed_edit,
    _find_stop_for_days,
    _splice_stop_days,
    _stop_day_numbers,
    _total_cost,
    apply_macro_edit,
    apply_stop_edit,
    classify_edit,
    respond_unsupported,
    route_after_classify,
    validate_edit,
)


def _activity(name: str, cost: float = 10.0) -> Activity:
    return Activity(
        name=name, description="d", category="sightseeing", estimated_cost_usd=cost, source="llm_estimate"
    )


def make_trip_request() -> TripRequest:
    return TripRequest(
        destination="Japan",
        duration_days=7,
        budget_usd=2000,
        traveler_count=2,
        interests=["food", "culture"],
        pace="moderate",
    )


def make_macro_plan() -> MacroPlan:
    return MacroPlan(
        stops=[
            MacroStop(city="Tokyo", country="Japan", days=3, rationale="r"),
            MacroStop(city="Kyoto", country="Japan", days=2, rationale="r"),
            MacroStop(city="Osaka", country="Japan", days=2, rationale="r"),
        ]
    )


def make_daily_itinerary(macro_plan: MacroPlan, cost: float = 10.0) -> DailyItinerary:
    days = []
    day_number = 1
    for stop in macro_plan.stops:
        for _ in range(stop.days):
            days.append(
                DayPlan(
                    day_number=day_number,
                    city=stop.city,
                    activities=[_activity(f"Day {day_number} activity", cost)],
                )
            )
            day_number += 1
    return DailyItinerary(days=days, estimated_total_cost_usd=_total_cost(days))


def make_edit_state(**overrides) -> dict:
    trip_request = make_trip_request()
    macro_plan = make_macro_plan()
    daily_itinerary = make_daily_itinerary(macro_plan)
    state = {
        "trip_request": trip_request,
        "macro_plan": macro_plan,
        "daily_itinerary": daily_itinerary,
        "edit_request": "some edit",
        "edit_scope": None,
        "target_day_numbers": None,
        "classification_note": None,
        "edit_retry_count": 0,
        "validation_feedback": None,
        "candidate_macro_plan": None,
        "candidate_daily_itinerary": None,
        "response_message": None,
    }
    state.update(overrides)
    return state


# --- 1. supported macro edit -------------------------------------------------


def test_apply_macro_edit_supported(monkeypatch):
    state = make_edit_state(edit_request="give Tokyo one more day and take it from Osaka")

    reallocated = MacroPlan(
        stops=[
            MacroStop(city="Tokyo", country="Japan", days=4, rationale="r"),
            MacroStop(city="Kyoto", country="Japan", days=2, rationale="r"),
            MacroStop(city="Osaka", country="Japan", days=1, rationale="r"),
        ]
    )

    class FakeMacroChain:
        def invoke(self, args):
            return reallocated

    monkeypatch.setattr(itinerary, "macro_chain", FakeMacroChain())

    def fake_plan_stop(trip_request, stop, extra_instruction=None):
        return StopItinerary(
            days=[StopDayContent(activities=[_activity(f"{stop.city} act")]) for _ in range(stop.days)]
        )

    monkeypatch.setattr(itinerary, "_plan_stop", fake_plan_stop)

    result = apply_macro_edit(state)

    assert result["candidate_macro_plan"] == reallocated
    days = result["candidate_daily_itinerary"].days
    assert [d.city for d in days] == ["Tokyo"] * 4 + ["Kyoto"] * 2 + ["Osaka"] * 1
    assert [d.day_number for d in days] == list(range(1, 8))


# --- 2. supported stop edit ---------------------------------------------------


def test_apply_stop_edit_supported(monkeypatch):
    macro_plan = make_macro_plan()
    daily_itinerary = make_daily_itinerary(macro_plan)
    state = make_edit_state(
        macro_plan=macro_plan,
        daily_itinerary=daily_itinerary,
        edit_request="less museums in Kyoto",
        edit_scope="stop",
        target_day_numbers=[4, 5],
    )

    new_stop_itinerary = StopItinerary(
        days=[
            StopDayContent(activities=[_activity("Nishiki Market", cost=15.0)]),
            StopDayContent(activities=[_activity("Arashiyama Bamboo Grove", cost=5.0)]),
        ]
    )
    monkeypatch.setattr(itinerary, "_plan_stop", lambda tr, stop, extra_instruction=None: new_stop_itinerary)

    result = apply_stop_edit(state)

    new_days = result["candidate_daily_itinerary"].days
    baseline_days = daily_itinerary.days

    # Untouched days are the exact same baseline objects.
    assert new_days[0] is baseline_days[0]
    assert new_days[1] is baseline_days[1]
    assert new_days[2] is baseline_days[2]
    assert new_days[5] is baseline_days[5]
    assert new_days[6] is baseline_days[6]

    # Targeted days (index 3, 4 -> day_number 4, 5) were replaced.
    assert new_days[3].day_number == 4
    assert new_days[3].city == "Kyoto"
    assert new_days[3].activities[0].name == "Nishiki Market"
    assert new_days[4].activities[0].name == "Arashiyama Bamboo Grove"

    assert result["candidate_macro_plan"] == macro_plan


# --- 3. unsupported request ---------------------------------------------------


def test_unsupported_request(monkeypatch):
    state = make_edit_state(edit_request="what's the deal with the JR pass?")

    classification = EditClassification(
        scope="unsupported", target_day_numbers=[], note="This is a question, not a change request."
    )

    class FakeClassifyChain:
        def invoke(self, args):
            return classification

    monkeypatch.setattr(itinerary, "edit_classify_chain", FakeClassifyChain())

    called = {"macro": False, "stop": False}
    monkeypatch.setattr(itinerary, "apply_macro_edit", lambda s: called.update(macro=True))
    monkeypatch.setattr(itinerary, "apply_stop_edit", lambda s: called.update(stop=True))

    result = classify_edit(state)
    state.update(result)

    assert state["edit_scope"] == "unsupported"
    assert route_after_classify(state) == "unsupported"
    assert called == {"macro": False, "stop": False}

    response = respond_unsupported(state)
    assert "JR pass" not in response["response_message"]
    assert "question, not a change request" in response["response_message"]


# --- 4 & 5. validation failure + retry, retry uses baseline not failed candidate ---


def test_validation_failure_triggers_retry_from_baseline(monkeypatch):
    state = make_edit_state(edit_request="rebalance the trip", edit_scope="macro")

    invalid_plan = MacroPlan(
        stops=[
            MacroStop(city="Tokyo", country="Japan", days=4, rationale="r"),
            MacroStop(city="Kyoto", country="Japan", days=2, rationale="r"),
            MacroStop(city="Osaka", country="Japan", days=2, rationale="r"),
        ]
    )  # sums to 8, not 7 -> invalid
    valid_plan = MacroPlan(
        stops=[
            MacroStop(city="Tokyo", country="Japan", days=4, rationale="r"),
            MacroStop(city="Kyoto", country="Japan", days=2, rationale="r"),
            MacroStop(city="Osaka", country="Japan", days=1, rationale="r"),
        ]
    )  # sums to 7 -> valid

    calls = []

    class FakeMacroChain:
        def invoke(self, args):
            calls.append(args)
            return invalid_plan if len(calls) == 1 else valid_plan

    monkeypatch.setattr(itinerary, "macro_chain", FakeMacroChain())

    def fake_plan_stop(trip_request, stop, extra_instruction=None):
        return StopItinerary(
            days=[StopDayContent(activities=[_activity(f"{stop.city} act")]) for _ in range(stop.days)]
        )

    monkeypatch.setattr(itinerary, "_plan_stop", fake_plan_stop)

    # Attempt 1
    state.update(apply_macro_edit(state))
    state.update(validate_edit(state))
    assert state["validation_feedback"] is not None
    assert state["edit_retry_count"] == 1
    assert "sum to 8" in state["validation_feedback"]

    # Baseline must remain untouched after the failed attempt.
    assert state["macro_plan"].stops[0].days == 3

    # Attempt 2 (retry)
    state.update(apply_macro_edit(state))
    state.update(validate_edit(state))
    assert state["validation_feedback"] is None

    assert len(calls) == 2
    # Both calls were built from the same baseline trip_request JSON - the
    # second call did not derive its input from the first (invalid) candidate.
    assert calls[0]["trip_request"] == calls[1]["trip_request"]
    assert calls[0]["trip_request"] == state["trip_request"].model_dump_json()
    # Only the feedback differs between attempts.
    assert calls[0]["feedback"] != calls[1]["feedback"]
    assert "Fix these issues" in calls[1]["feedback"]


# --- 6. target day resolution -------------------------------------------------


def test_find_stop_for_days():
    macro_plan = make_macro_plan()  # Tokyo 1-3, Kyoto 4-5, Osaka 6-7

    assert _find_stop_for_days(macro_plan, [4, 5]).city == "Kyoto"
    assert _find_stop_for_days(macro_plan, [1, 2, 3]).city == "Tokyo"
    assert _find_stop_for_days(macro_plan, [6, 7]).city == "Osaka"
    assert _find_stop_for_days(macro_plan, [4]).city == "Kyoto"

    with pytest.raises(ValueError):
        _find_stop_for_days(macro_plan, [3, 4])  # spans Tokyo and Kyoto


# --- 7. cost recomputation -----------------------------------------------------


def test_total_cost():
    days = [
        DayPlan(day_number=1, city="Tokyo", activities=[_activity("a", 10.0), _activity("b", None)]),
        DayPlan(day_number=2, city="Tokyo", activities=[_activity("c", 5.0)]),
    ]
    assert _total_cost(days) == 15.0

    no_cost_days = [DayPlan(day_number=1, city="Tokyo", activities=[_activity("a", None)])]
    assert _total_cost(no_cost_days) is None


def test_cost_recomputation_after_splice():
    macro_plan = make_macro_plan()
    baseline = make_daily_itinerary(macro_plan, cost=10.0)  # 7 activities x $10 = $70
    stop = macro_plan.stops[1]  # Kyoto, days 4-5

    new_days = _splice_stop_days(
        baseline.days,
        [4, 5],
        stop,
        [
            StopDayContent(activities=[_activity("x", 20.0)]),
            StopDayContent(activities=[_activity("y", 20.0)]),
        ],
    )
    total = _total_cost(new_days)
    # 5 untouched activities x $10 + 2 replaced activities x $20 = $90
    assert total == 90.0


# --- 8. end-to-end edit through the graph --------------------------------------


def test_edit_graph_end_to_end(monkeypatch):
    macro_plan = make_macro_plan()
    daily_itinerary = make_daily_itinerary(macro_plan)

    classification = EditClassification(
        scope="stop", target_day_numbers=[4, 5], note="Reduce museums in Kyoto."
    )

    class FakeClassifyChain:
        def invoke(self, args):
            return classification

    monkeypatch.setattr(itinerary, "edit_classify_chain", FakeClassifyChain())

    new_stop_itinerary = StopItinerary(
        days=[
            StopDayContent(activities=[_activity("Nishiki Market", cost=15.0)]),
            StopDayContent(activities=[_activity("Arashiyama Bamboo Grove", cost=5.0)]),
        ]
    )
    monkeypatch.setattr(itinerary, "_plan_stop", lambda tr, stop, extra_instruction=None: new_stop_itinerary)

    initial_state = make_edit_state(
        macro_plan=macro_plan,
        daily_itinerary=daily_itinerary,
        edit_request="less museums in Kyoto",
    )

    result = itinerary.edit_graph.invoke(initial_state)

    assert result["edit_scope"] == "stop"
    assert result["validation_feedback"] is None
    assert result["candidate_daily_itinerary"] is not None
    kyoto_days = [d for d in result["candidate_daily_itinerary"].days if d.day_number in (4, 5)]
    assert kyoto_days[0].activities[0].name == "Nishiki Market"
    assert kyoto_days[1].activities[0].name == "Arashiyama Bamboo Grove"


# --- 9. committing the candidate back into the CLI's plan state ----------------


def test_apply_committed_edit_unsupported():
    macro_plan = make_macro_plan()
    daily_itinerary = make_daily_itinerary(macro_plan)
    result = {"edit_scope": "unsupported", "response_message": "can't do that"}

    new_macro, new_daily, message = _apply_committed_edit(macro_plan, daily_itinerary, result)

    assert new_macro is macro_plan
    assert new_daily is daily_itinerary
    assert message == "can't do that"


def test_apply_committed_edit_supported():
    macro_plan = make_macro_plan()
    daily_itinerary = make_daily_itinerary(macro_plan)
    candidate_macro = make_macro_plan()
    candidate_daily = make_daily_itinerary(candidate_macro)
    result = {
        "edit_scope": "stop",
        "candidate_macro_plan": candidate_macro,
        "candidate_daily_itinerary": candidate_daily,
    }

    new_macro, new_daily, message = _apply_committed_edit(macro_plan, daily_itinerary, result)

    assert new_macro is candidate_macro
    assert new_daily is candidate_daily
    assert message is None


# --- classify_edit: valid resolution and invalid-target downgrade --------------


def test_classify_edit_resolves_stop_scope(monkeypatch):
    state = make_edit_state(edit_request="less museums in Kyoto")

    classification = EditClassification(
        scope="stop", target_day_numbers=[4, 5], note="Reduce museums in Kyoto."
    )
    captured = {}

    class FakeClassifyChain:
        def invoke(self, args):
            captured.update(args)
            return classification

    monkeypatch.setattr(itinerary, "edit_classify_chain", FakeClassifyChain())

    result = classify_edit(state)

    assert result["edit_scope"] == "stop"
    assert result["target_day_numbers"] == [4, 5]
    assert result["classification_note"] == "Reduce museums in Kyoto."
    assert "day 4: Kyoto" in captured["daily_summary"]
    assert "day 5: Kyoto" in captured["daily_summary"]


def test_classify_edit_downgrades_empty_target(monkeypatch):
    state = make_edit_state(edit_request="less museums in Kyoto")

    classification = EditClassification(scope="stop", target_day_numbers=[], note="Reduce museums.")

    class FakeClassifyChain:
        def invoke(self, args):
            return classification

    monkeypatch.setattr(itinerary, "edit_classify_chain", FakeClassifyChain())

    result = classify_edit(state)

    assert result["edit_scope"] == "unsupported"
    assert "no target day(s)" in result["classification_note"]


def test_classify_edit_downgrades_out_of_range_target(monkeypatch):
    state = make_edit_state(edit_request="more food on day 99")

    classification = EditClassification(scope="stop", target_day_numbers=[99], note="Add food.")

    class FakeClassifyChain:
        def invoke(self, args):
            return classification

    monkeypatch.setattr(itinerary, "edit_classify_chain", FakeClassifyChain())

    result = classify_edit(state)

    assert result["edit_scope"] == "unsupported"
    assert "99" in result["classification_note"]
    assert "don't exist" in result["classification_note"]


def test_classify_edit_downgrades_multi_stop_target(monkeypatch):
    state = make_edit_state(edit_request="rework days 3 and 4")

    classification = EditClassification(scope="stop", target_day_numbers=[3, 4], note="Rework.")

    class FakeClassifyChain:
        def invoke(self, args):
            return classification

    monkeypatch.setattr(itinerary, "edit_classify_chain", FakeClassifyChain())

    result = classify_edit(state)

    assert result["edit_scope"] == "unsupported"
    assert "span more than one stop" in result["classification_note"]


# --- apply_stop_edit: whole-stop widening ---------------------------------------


def test_apply_stop_edit_widens_to_full_stop(monkeypatch):
    macro_plan = make_macro_plan()
    daily_itinerary = make_daily_itinerary(macro_plan)
    # Simulates "more food on day 4": classification targets only day 4, but
    # Kyoto (the owning stop) spans days 4-5.
    state = make_edit_state(
        macro_plan=macro_plan,
        daily_itinerary=daily_itinerary,
        edit_request="more food on day 4",
        edit_scope="stop",
        target_day_numbers=[4],
    )

    new_stop_itinerary = StopItinerary(
        days=[
            StopDayContent(activities=[_activity("Food day 4", cost=10.0)]),
            StopDayContent(activities=[_activity("Food day 5", cost=10.0)]),
        ]
    )
    monkeypatch.setattr(itinerary, "_plan_stop", lambda tr, stop, extra_instruction=None: new_stop_itinerary)

    result = apply_stop_edit(state)

    new_days = result["candidate_daily_itinerary"].days
    baseline_days = daily_itinerary.days

    # Both Kyoto days (4 and 5) were regenerated, not just day 4.
    assert new_days[3].activities[0].name == "Food day 4"
    assert new_days[4].activities[0].name == "Food day 5"
    # Everything outside the owning stop is untouched (identity, not just equality).
    assert new_days[0] is baseline_days[0]
    assert new_days[1] is baseline_days[1]
    assert new_days[2] is baseline_days[2]
    assert new_days[5] is baseline_days[5]
    assert new_days[6] is baseline_days[6]


def test_stop_day_numbers():
    macro_plan = make_macro_plan()  # Tokyo 1-3, Kyoto 4-5, Osaka 6-7
    kyoto = macro_plan.stops[1]
    assert _stop_day_numbers(macro_plan, kyoto) == [4, 5]


# --- apply_macro_edit: baseline plan reaches the prompt -------------------------


def test_apply_macro_edit_includes_current_plan_in_prompt(monkeypatch):
    macro_plan = make_macro_plan()
    state = make_edit_state(
        macro_plan=macro_plan,
        edit_request="give Tokyo one more day and take it from Osaka",
    )

    captured = {}
    reallocated = MacroPlan(
        stops=[
            MacroStop(city="Tokyo", country="Japan", days=4, rationale="r"),
            MacroStop(city="Kyoto", country="Japan", days=2, rationale="r"),
            MacroStop(city="Osaka", country="Japan", days=1, rationale="r"),
        ]
    )

    class FakeMacroChain:
        def invoke(self, args):
            captured.update(args)
            return reallocated

    monkeypatch.setattr(itinerary, "macro_chain", FakeMacroChain())
    monkeypatch.setattr(
        itinerary,
        "_plan_stop",
        lambda tr, stop, extra_instruction=None: StopItinerary(
            days=[StopDayContent(activities=[_activity(f"{stop.city} act")]) for _ in range(stop.days)]
        ),
    )

    apply_macro_edit(state)

    assert "Tokyo (3d)" in captured["current_plan"]
    assert "Kyoto (2d)" in captured["current_plan"]
    assert "Osaka (2d)" in captured["current_plan"]


# --- edit_graph: real routing through a failure -> retry -> success cycle ------


def test_edit_graph_retries_on_validation_failure(monkeypatch):
    macro_plan = make_macro_plan()
    daily_itinerary = make_daily_itinerary(macro_plan)

    classification = EditClassification(scope="macro", target_day_numbers=[], note="Rebalance days.")

    class FakeClassifyChain:
        def invoke(self, args):
            return classification

    monkeypatch.setattr(itinerary, "edit_classify_chain", FakeClassifyChain())

    invalid_plan = MacroPlan(
        stops=[
            MacroStop(city="Tokyo", country="Japan", days=4, rationale="r"),
            MacroStop(city="Kyoto", country="Japan", days=2, rationale="r"),
            MacroStop(city="Osaka", country="Japan", days=2, rationale="r"),
        ]
    )  # sums to 8, invalid
    valid_plan = MacroPlan(
        stops=[
            MacroStop(city="Tokyo", country="Japan", days=4, rationale="r"),
            MacroStop(city="Kyoto", country="Japan", days=2, rationale="r"),
            MacroStop(city="Osaka", country="Japan", days=1, rationale="r"),
        ]
    )  # sums to 7, valid

    macro_calls = []

    class FakeMacroChain:
        def invoke(self, args):
            macro_calls.append(args)
            return invalid_plan if len(macro_calls) == 1 else valid_plan

    monkeypatch.setattr(itinerary, "macro_chain", FakeMacroChain())
    monkeypatch.setattr(
        itinerary,
        "_plan_stop",
        lambda tr, stop, extra_instruction=None: StopItinerary(
            days=[StopDayContent(activities=[_activity(f"{stop.city} act")]) for _ in range(stop.days)]
        ),
    )

    result = itinerary.edit_graph.invoke(
        make_edit_state(
            macro_plan=macro_plan, daily_itinerary=daily_itinerary, edit_request="rebalance the trip"
        )
    )

    assert len(macro_calls) == 2  # apply_macro_edit ran twice: initial + one retry
    assert result["edit_retry_count"] == 1
    assert result["validation_feedback"] is None
    assert result["candidate_macro_plan"] == valid_plan
    assert sum(s.days for s in result["candidate_macro_plan"].stops) == 7

    # Prove the second attempt actually received the validation feedback
    # produced by the first failed candidate - not just that it was called
    # twice, but that the retry mechanism propagated the failure reason.
    first_feedback = macro_calls[0]["feedback"]
    second_feedback = macro_calls[1]["feedback"]
    assert first_feedback != second_feedback
    assert "sum to 8" in second_feedback
    assert "Fix these issues" in second_feedback

    # The baseline context sent to the model must be identical across both
    # attempts - the retry must not have derived it from the failed candidate.
    assert macro_calls[0]["current_plan"] == macro_calls[1]["current_plan"]
    assert macro_calls[0]["current_plan"] == "Tokyo (3d); Kyoto (2d); Osaka (2d)"

    # The graph's own baseline state must still be the exact original object,
    # never overwritten by either attempt's candidate.
    assert result["macro_plan"] is macro_plan


# --- edit_graph: invalid target is a safe no-op, never reaches generation ------


def test_edit_graph_invalid_target_is_noop(monkeypatch):
    macro_plan = make_macro_plan()
    daily_itinerary = make_daily_itinerary(macro_plan)

    classification = EditClassification(scope="stop", target_day_numbers=[], note="Some stop edit.")

    class FakeClassifyChain:
        def invoke(self, args):
            return classification

    monkeypatch.setattr(itinerary, "edit_classify_chain", FakeClassifyChain())

    def fail_if_called(*args, **kwargs):
        pytest.fail("generation should not be reached for an invalid target")

    class FailingMacroChain:
        def invoke(self, args):
            fail_if_called()

    monkeypatch.setattr(itinerary, "macro_chain", FailingMacroChain())
    monkeypatch.setattr(itinerary, "_plan_stop", fail_if_called)

    result = itinerary.edit_graph.invoke(
        make_edit_state(
            macro_plan=macro_plan, daily_itinerary=daily_itinerary, edit_request="more food on day 4"
        )
    )

    assert result["edit_scope"] == "unsupported"
    assert result["response_message"] is not None
    assert "no target day(s)" in result["response_message"]
    assert result["candidate_macro_plan"] is None
    assert result["candidate_daily_itinerary"] is None

    new_macro, new_daily, message = _apply_committed_edit(macro_plan, daily_itinerary, result)
    assert new_macro is macro_plan
    assert new_daily is daily_itinerary
    assert message == result["response_message"]


# --- edit_graph: partial target is widened to the full stop, through the real graph ---


def test_edit_graph_widens_partial_target_to_full_stop(monkeypatch):
    macro_plan = make_macro_plan()  # Tokyo 1-3, Kyoto 4-5, Osaka 6-7
    daily_itinerary = make_daily_itinerary(macro_plan)

    # Classifier identifies only day 4 ("more food on day 4"), even though
    # Kyoto (the owning stop) spans days 4-5.
    classification = EditClassification(
        scope="stop", target_day_numbers=[4], note="Add more food on day 4."
    )

    class FakeClassifyChain:
        def invoke(self, args):
            return classification

    monkeypatch.setattr(itinerary, "edit_classify_chain", FakeClassifyChain())

    new_stop_itinerary = StopItinerary(
        days=[
            StopDayContent(activities=[_activity("Food day 4", cost=10.0)]),
            StopDayContent(activities=[_activity("Food day 5", cost=10.0)]),
        ]
    )
    monkeypatch.setattr(itinerary, "_plan_stop", lambda tr, stop, extra_instruction=None: new_stop_itinerary)

    result = itinerary.edit_graph.invoke(
        make_edit_state(
            macro_plan=macro_plan, daily_itinerary=daily_itinerary, edit_request="more food on day 4"
        )
    )

    assert result["edit_scope"] == "stop"
    assert result["validation_feedback"] is None

    new_days = result["candidate_daily_itinerary"].days
    baseline_days = daily_itinerary.days

    # Both Kyoto days (4 and 5) were replaced, not just the classifier's
    # single targeted day 4 - proving the widening happens through the real
    # compiled graph, not just when apply_stop_edit is called directly.
    assert new_days[3].day_number == 4
    assert new_days[3].activities[0].name == "Food day 4"
    assert new_days[4].day_number == 5
    assert new_days[4].activities[0].name == "Food day 5"

    # Days outside the targeted stop remain the exact same baseline objects.
    assert new_days[0] is baseline_days[0]
    assert new_days[1] is baseline_days[1]
    assert new_days[2] is baseline_days[2]
    assert new_days[5] is baseline_days[5]
    assert new_days[6] is baseline_days[6]
