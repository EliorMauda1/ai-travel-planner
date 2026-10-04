"""Phase 4 (+ cross-stop/content-edit fix): tests for the conversational
plan-editing graph.

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
    _build_daily_itinerary,
    _plan_stop_exact,
    _resolve_stops,
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


def _activity(name: str, cost: float = 10.0, category: str = "sightseeing") -> Activity:
    return Activity(
        name=name, description="d", category=category, estimated_cost_usd=cost, source="llm_estimate"
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
        "active_excluded_categories": [],
        "effective_excluded_categories": [],
        "edit_scope": None,
        "target_stops": None,
        "classification_note": None,
        "edit_retry_count": 0,
        "validation_feedback": None,
        "blocking_violation": None,
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


def test_apply_macro_edit_passes_instruction_to_daily_regeneration(monkeypatch):
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

    captured_instructions = []

    def fake_plan_stop(trip_request, stop, extra_instruction=None):
        captured_instructions.append(extra_instruction)
        return StopItinerary(
            days=[StopDayContent(activities=[_activity(f"{stop.city} act")]) for _ in range(stop.days)]
        )

    monkeypatch.setattr(itinerary, "_plan_stop", fake_plan_stop)

    apply_macro_edit(state)

    assert len(captured_instructions) == 3  # one call per regenerated stop
    assert all(
        instr is not None and "give Tokyo one more day" in instr for instr in captured_instructions
    )


# --- 2. supported stop edit ---------------------------------------------------


def test_apply_stop_edit_supported(monkeypatch):
    macro_plan = make_macro_plan()
    daily_itinerary = make_daily_itinerary(macro_plan)
    state = make_edit_state(
        macro_plan=macro_plan,
        daily_itinerary=daily_itinerary,
        edit_request="less museums in Kyoto",
        edit_scope="stop",
        target_stops=["Kyoto"],
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


def test_apply_stop_edit_regenerates_whole_stop_for_resolved_target(monkeypatch):
    """apply_stop_edit does not perform any day->stop resolution itself - it
    receives an already-resolved target_stops=['Kyoto'] (the shape a
    classifier would produce for a day-specific request like 'more food on
    day 4') and regenerates that whole stop, not just one day within it."""
    macro_plan = make_macro_plan()  # Tokyo 1-3, Kyoto 4-5, Osaka 6-7
    daily_itinerary = make_daily_itinerary(macro_plan)
    state = make_edit_state(
        macro_plan=macro_plan,
        daily_itinerary=daily_itinerary,
        edit_request="more food on day 4",
        edit_scope="stop",
        target_stops=["Kyoto"],
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
    # Everything outside Kyoto is untouched (identity, not just equality).
    assert new_days[0] is baseline_days[0]
    assert new_days[1] is baseline_days[1]
    assert new_days[2] is baseline_days[2]
    assert new_days[5] is baseline_days[5]
    assert new_days[6] is baseline_days[6]


def test_apply_stop_edit_multiple_named_stops(monkeypatch):
    macro_plan = make_macro_plan()  # Tokyo 1-3, Kyoto 4-5, Osaka 6-7
    daily_itinerary = make_daily_itinerary(macro_plan)
    state = make_edit_state(
        macro_plan=macro_plan,
        daily_itinerary=daily_itinerary,
        edit_request="exclude food in Kyoto and Osaka, leave Tokyo alone",
        edit_scope="stop",
        target_stops=["Kyoto", "Osaka"],
    )

    def fake_plan_stop(trip_request, stop, extra_instruction=None):
        return StopItinerary(
            days=[
                StopDayContent(activities=[_activity(f"{stop.city} new act")])
                for _ in range(stop.days)
            ]
        )

    monkeypatch.setattr(itinerary, "_plan_stop", fake_plan_stop)

    result = apply_stop_edit(state)

    new_days = result["candidate_daily_itinerary"].days
    baseline_days = daily_itinerary.days

    # Tokyo (days 1-3), not targeted, is untouched - identical baseline objects.
    assert new_days[0] is baseline_days[0]
    assert new_days[1] is baseline_days[1]
    assert new_days[2] is baseline_days[2]

    # Kyoto (4-5) and Osaka (6-7) were both regenerated.
    assert new_days[3].activities[0].name == "Kyoto new act"
    assert new_days[4].activities[0].name == "Kyoto new act"
    assert new_days[5].activities[0].name == "Osaka new act"
    assert new_days[6].activities[0].name == "Osaka new act"

    assert result["candidate_macro_plan"] == macro_plan


def test_apply_stop_edit_all_named_stops(monkeypatch):
    macro_plan = make_macro_plan()
    daily_itinerary = make_daily_itinerary(macro_plan)
    state = make_edit_state(
        macro_plan=macro_plan,
        daily_itinerary=daily_itinerary,
        edit_request="exclude food everywhere, kosher only",
        edit_scope="stop",
        target_stops=["Tokyo", "Kyoto", "Osaka"],
    )

    def fake_plan_stop(trip_request, stop, extra_instruction=None):
        return StopItinerary(
            days=[
                StopDayContent(activities=[_activity(f"{stop.city} sightseeing")])
                for _ in range(stop.days)
            ]
        )

    monkeypatch.setattr(itinerary, "_plan_stop", fake_plan_stop)

    result = apply_stop_edit(state)

    new_days = result["candidate_daily_itinerary"].days
    baseline_days = daily_itinerary.days

    # Every day was regenerated - none are the original baseline objects.
    assert all(new is not old for new, old in zip(new_days, baseline_days))
    assert [d.activities[0].name for d in new_days] == [
        "Tokyo sightseeing",
        "Tokyo sightseeing",
        "Tokyo sightseeing",
        "Kyoto sightseeing",
        "Kyoto sightseeing",
        "Osaka sightseeing",
        "Osaka sightseeing",
    ]
    assert result["candidate_macro_plan"] is macro_plan


# --- 3. unsupported request ---------------------------------------------------


def test_unsupported_request(monkeypatch):
    state = make_edit_state(edit_request="what's the deal with the JR pass?")

    classification = EditClassification(
        scope="unsupported", target_stops=[], note="This is a question, not a change request."
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


# --- 6. target stop resolution -------------------------------------------------


def test_resolve_stops():
    macro_plan = make_macro_plan()  # Tokyo, Kyoto, Osaka (in that order)

    assert [s.city for s in _resolve_stops(macro_plan, ["Kyoto"])] == ["Kyoto"]
    # Order follows the trip's own order, not the input list's order.
    assert [s.city for s in _resolve_stops(macro_plan, ["Osaka", "Tokyo"])] == ["Tokyo", "Osaka"]
    # Case-insensitive.
    assert [s.city for s in _resolve_stops(macro_plan, ["kyoto"])] == ["Kyoto"]
    # Duplicate input names don't produce duplicate stops.
    assert [s.city for s in _resolve_stops(macro_plan, ["Kyoto", "Kyoto"])] == ["Kyoto"]
    # Unknown names are simply ignored (validation happens separately, in
    # _validate_stop_targets / classify_edit).
    assert _resolve_stops(macro_plan, ["Atlantis"]) == []


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
        scope="stop", target_stops=["Kyoto"], note="Reduce museums in Kyoto."
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


def test_edit_graph_multiple_named_stops(monkeypatch):
    macro_plan = make_macro_plan()  # Tokyo 1-3, Kyoto 4-5, Osaka 6-7
    daily_itinerary = make_daily_itinerary(macro_plan)

    classification = EditClassification(
        scope="stop",
        target_stops=["Kyoto", "Osaka"],
        note="Exclude food in Kyoto and Osaka.",
    )

    class FakeClassifyChain:
        def invoke(self, args):
            return classification

    monkeypatch.setattr(itinerary, "edit_classify_chain", FakeClassifyChain())

    def fake_plan_stop(trip_request, stop, extra_instruction=None):
        return StopItinerary(
            days=[
                StopDayContent(activities=[_activity(f"{stop.city} sightseeing")])
                for _ in range(stop.days)
            ]
        )

    monkeypatch.setattr(itinerary, "_plan_stop", fake_plan_stop)

    result = itinerary.edit_graph.invoke(
        make_edit_state(
            macro_plan=macro_plan,
            daily_itinerary=daily_itinerary,
            edit_request="exclude food in Kyoto and Osaka, leave Tokyo alone",
        )
    )

    assert result["edit_scope"] == "stop"
    assert result["validation_feedback"] is None

    new_days = result["candidate_daily_itinerary"].days
    baseline_days = daily_itinerary.days

    # Tokyo (days 1-3) untouched, through the real compiled graph.
    assert new_days[0] is baseline_days[0]
    assert new_days[1] is baseline_days[1]
    assert new_days[2] is baseline_days[2]
    # Kyoto and Osaka regenerated.
    assert new_days[3].activities[0].name == "Kyoto sightseeing"
    assert new_days[4].activities[0].name == "Kyoto sightseeing"
    assert new_days[5].activities[0].name == "Osaka sightseeing"
    assert new_days[6].activities[0].name == "Osaka sightseeing"


def test_edit_graph_represents_day_specific_request_as_stop_target(monkeypatch):
    """With a stubbed classifier returning target_stops=['Kyoto'] for a
    day-specific request ('more food on day 4'), this proves: (a) a
    day-specific request CAN be represented as a single stop target, (b) the
    graph then regenerates the whole targeted stop, and (c) unrelated stops'
    days stay identical by object identity. It does NOT prove that the real
    LLM classifier correctly performs the day->stop semantic resolution
    itself - that happens only when the live model is exercised, not here."""
    macro_plan = make_macro_plan()  # Tokyo 1-3, Kyoto 4-5, Osaka 6-7
    daily_itinerary = make_daily_itinerary(macro_plan)

    classification = EditClassification(
        scope="stop", target_stops=["Kyoto"], note="Add more food on day 4."
    )

    class FakeClassifyChain:
        def invoke(self, args):
            # Sanity: the classifier is at least given the day->city mapping
            # it would need in order to perform that resolution itself.
            assert "day 4: Kyoto" in args["daily_summary"]
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

    # Both Kyoto days (4 and 5) were replaced, not just the day the user
    # literally mentioned.
    assert new_days[3].day_number == 4
    assert new_days[3].activities[0].name == "Food day 4"
    assert new_days[4].day_number == 5
    assert new_days[4].activities[0].name == "Food day 5"

    # Days outside Kyoto remain the exact same baseline objects.
    assert new_days[0] is baseline_days[0]
    assert new_days[1] is baseline_days[1]
    assert new_days[2] is baseline_days[2]
    assert new_days[5] is baseline_days[5]
    assert new_days[6] is baseline_days[6]


# --- 9. committing the candidate back into the CLI's plan state ----------------


def test_apply_committed_edit_unsupported():
    macro_plan = make_macro_plan()
    daily_itinerary = make_daily_itinerary(macro_plan)
    result = {"edit_scope": "unsupported", "response_message": "can't do that"}

    new_macro, new_daily, new_excl, message = _apply_committed_edit(
        macro_plan, daily_itinerary, [], result
    )

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
        "effective_excluded_categories": ["food"],
    }

    new_macro, new_daily, new_excl, message = _apply_committed_edit(
        macro_plan, daily_itinerary, [], result
    )

    assert new_macro is candidate_macro
    assert new_daily is candidate_daily
    assert new_excl == ["food"]
    assert message is None


def test_apply_committed_edit_blocks_blocking_violation():
    macro_plan = make_macro_plan()
    daily_itinerary = make_daily_itinerary(macro_plan)
    candidate_macro = make_macro_plan()
    candidate_daily = make_daily_itinerary(candidate_macro)
    result = {
        "edit_scope": "stop",
        "candidate_macro_plan": candidate_macro,
        "candidate_daily_itinerary": candidate_daily,
        "blocking_violation": "day 4 contains a 'food' activity (Ramen shop) "
        "despite the exclusion request",
    }

    new_macro, new_daily, new_excl, message = _apply_committed_edit(
        macro_plan, daily_itinerary, [], result
    )

    # The violating candidate must NOT be committed - baseline preserved by identity.
    assert new_macro is macro_plan
    assert new_daily is daily_itinerary
    assert message is not None
    assert "Ramen shop" in message
    assert "safely apply" in message


def test_apply_committed_edit_blocks_day_count_violation():
    macro_plan = make_macro_plan()
    daily_itinerary = make_daily_itinerary(macro_plan)
    candidate_macro = make_macro_plan()
    candidate_daily = make_daily_itinerary(candidate_macro)
    result = {
        "edit_scope": "macro",
        "candidate_macro_plan": candidate_macro,
        "candidate_daily_itinerary": candidate_daily,
        "blocking_violation": "daily itinerary has 8 day(s) but the macro plan's "
        "stops sum to 7 day(s)",
    }

    new_macro, new_daily, new_excl, message = _apply_committed_edit(
        macro_plan, daily_itinerary, [], result
    )

    # The incoherent candidate must NOT be committed - baseline preserved by identity.
    assert new_macro is macro_plan
    assert new_daily is daily_itinerary
    assert message is not None
    assert "safely apply" in message
    # The message must be generic, not specific to satisfying a preference -
    # it covers both hard-exclusion and structural-integrity violations.
    assert "satisfy that request" not in message


# --- classify_edit: valid resolution and invalid-target downgrade --------------


def test_classify_edit_resolves_stop_scope(monkeypatch):
    state = make_edit_state(edit_request="less museums in Kyoto")

    classification = EditClassification(
        scope="stop", target_stops=["Kyoto"], note="Reduce museums in Kyoto."
    )
    captured = {}

    class FakeClassifyChain:
        def invoke(self, args):
            captured.update(args)
            return classification

    monkeypatch.setattr(itinerary, "edit_classify_chain", FakeClassifyChain())

    result = classify_edit(state)

    assert result["edit_scope"] == "stop"
    assert result["target_stops"] == ["Kyoto"]
    assert result["classification_note"] == "Reduce museums in Kyoto."
    assert "day 4: Kyoto" in captured["daily_summary"]
    assert "day 5: Kyoto" in captured["daily_summary"]


def test_classify_edit_adds_durable_exclusion(monkeypatch):
    state = make_edit_state(edit_request="exclude food entirely, kosher only")

    classification = EditClassification(
        scope="stop",
        target_stops=["Tokyo", "Kyoto", "Osaka"],
        add_excluded_categories=["food"],
        note="Exclude food across the whole trip.",
    )

    class FakeClassifyChain:
        def invoke(self, args):
            return classification

    monkeypatch.setattr(itinerary, "edit_classify_chain", FakeClassifyChain())

    result = classify_edit(state)

    assert result["edit_scope"] == "stop"
    assert result["target_stops"] == ["Tokyo", "Kyoto", "Osaka"]
    assert result["effective_excluded_categories"] == ["food"]


def test_classify_edit_downgrades_empty_target(monkeypatch):
    state = make_edit_state(edit_request="less museums in Kyoto")

    classification = EditClassification(scope="stop", target_stops=[], note="Reduce museums.")

    class FakeClassifyChain:
        def invoke(self, args):
            return classification

    monkeypatch.setattr(itinerary, "edit_classify_chain", FakeClassifyChain())

    result = classify_edit(state)

    assert result["edit_scope"] == "unsupported"
    assert "no target stop(s)" in result["classification_note"]


def test_classify_edit_downgrades_unknown_stop_name(monkeypatch):
    state = make_edit_state(edit_request="less museums in Atlantis")

    classification = EditClassification(
        scope="stop", target_stops=["Atlantis"], note="Reduce museums."
    )

    class FakeClassifyChain:
        def invoke(self, args):
            return classification

    monkeypatch.setattr(itinerary, "edit_classify_chain", FakeClassifyChain())

    result = classify_edit(state)

    assert result["edit_scope"] == "unsupported"
    assert "Atlantis" in result["classification_note"]
    assert "don't match any stop" in result["classification_note"]


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

    classification = EditClassification(scope="macro", target_stops=[], note="Rebalance days.")

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

    classification = EditClassification(scope="stop", target_stops=[], note="Some stop edit.")

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
            macro_plan=macro_plan, daily_itinerary=daily_itinerary, edit_request="more food everywhere"
        )
    )

    assert result["edit_scope"] == "unsupported"
    assert result["response_message"] is not None
    assert "no target stop(s)" in result["response_message"]
    assert result["candidate_macro_plan"] is None
    assert result["candidate_daily_itinerary"] is None

    new_macro, new_daily, new_excl, message = _apply_committed_edit(
        macro_plan, daily_itinerary, [], result
    )
    assert new_macro is macro_plan
    assert new_daily is daily_itinerary
    assert message == result["response_message"]


# --- validate_edit: excluded_categories hard-constraint check (scoped to targets) ---


def test_validate_edit_detects_excluded_category_violation_in_targeted_stop():
    macro_plan = make_macro_plan()  # Tokyo 1-3, Kyoto 4-5, Osaka 6-7
    daily_itinerary = make_daily_itinerary(macro_plan)
    daily_itinerary.days[3].activities.append(_activity("Ramen shop", category="food"))

    state = make_edit_state(
        macro_plan=macro_plan,
        target_stops=["Kyoto"],
        effective_excluded_categories=["food"],
        edit_scope="stop",
        candidate_macro_plan=macro_plan,
        candidate_daily_itinerary=daily_itinerary,
    )

    result = validate_edit(state)

    assert result["validation_feedback"] is not None
    assert "food" in result["validation_feedback"]
    assert result["edit_retry_count"] == 1
    assert result["blocking_violation"] is not None
    assert "Ramen shop" in result["blocking_violation"]


def test_validate_edit_ignores_excluded_category_violation_outside_targeted_stops():
    macro_plan = make_macro_plan()  # Tokyo 1-3, Kyoto 4-5, Osaka 6-7
    daily_itinerary = make_daily_itinerary(macro_plan)
    # Pre-existing food activity in Osaka, which this edit does NOT target.
    daily_itinerary.days[5].activities.append(_activity("Takoyaki stand", category="food"))

    state = make_edit_state(
        macro_plan=macro_plan,
        target_stops=["Tokyo", "Kyoto"],  # Osaka not targeted
        effective_excluded_categories=["food"],
        edit_scope="stop",
        candidate_macro_plan=macro_plan,
        candidate_daily_itinerary=daily_itinerary,
    )

    result = validate_edit(state)

    assert result["validation_feedback"] is None
    assert result["blocking_violation"] is None


def test_validate_edit_passes_when_excluded_category_absent():
    macro_plan = make_macro_plan()
    daily_itinerary = make_daily_itinerary(macro_plan)  # default category "sightseeing"

    state = make_edit_state(
        macro_plan=macro_plan,
        target_stops=["Kyoto"],
        effective_excluded_categories=["food"],
        edit_scope="stop",
        candidate_macro_plan=macro_plan,
        candidate_daily_itinerary=daily_itinerary,
    )

    result = validate_edit(state)

    assert result["validation_feedback"] is None
    assert result["blocking_violation"] is None


def test_validate_edit_structural_failure_alone_leaves_blocking_violation_none():
    macro_plan = make_macro_plan()
    bad_macro_plan = MacroPlan(
        stops=[
            MacroStop(city="Tokyo", country="Japan", days=3, rationale="r"),
            MacroStop(city="Kyoto", country="Japan", days=2, rationale="r"),
            MacroStop(city="Osaka", country="Japan", days=3, rationale="r"),
        ]
    )  # sums to 8, not 7 -> structural failure (duration mismatch) only.
    # Built from bad_macro_plan (not the original 7-day macro_plan) so its
    # day count (8) matches bad_macro_plan's own total exactly - isolating
    # this test to ONLY the macro-sum-vs-duration_days problem, with no
    # accompanying daily-vs-macro day-count mismatch of its own.
    daily_itinerary = make_daily_itinerary(bad_macro_plan)

    state = make_edit_state(
        macro_plan=macro_plan,
        edit_scope="macro",
        candidate_macro_plan=bad_macro_plan,
        candidate_daily_itinerary=daily_itinerary,
    )

    result = validate_edit(state)

    assert result["validation_feedback"] is not None
    assert "sum to 8" in result["validation_feedback"]
    assert result["blocking_violation"] is None


# --- edit_graph: excluded_categories retry + hard-constraint commit-gating ------


def test_edit_graph_retries_stop_edit_on_excluded_category_violation(monkeypatch):
    macro_plan = make_macro_plan()
    daily_itinerary = make_daily_itinerary(macro_plan)

    # The exclusion is already persisted (no delta this turn), so no all-stop
    # expansion fires and the edit stays scoped to Kyoto.
    classification = EditClassification(
        scope="stop",
        target_stops=["Kyoto"],
        note="Rework Kyoto.",
    )

    class FakeClassifyChain:
        def invoke(self, args):
            return classification

    monkeypatch.setattr(itinerary, "edit_classify_chain", FakeClassifyChain())

    violating_itinerary = StopItinerary(
        days=[
            StopDayContent(activities=[_activity("Ramen shop", category="food")]),
            StopDayContent(activities=[_activity("Temple visit")]),
        ]
    )
    clean_itinerary = StopItinerary(
        days=[
            StopDayContent(activities=[_activity("Temple visit")]),
            StopDayContent(activities=[_activity("Garden walk", category="nature")]),
        ]
    )

    plan_stop_instructions = []

    def fake_plan_stop(trip_request, stop, extra_instruction=None):
        plan_stop_instructions.append(extra_instruction)
        return violating_itinerary if len(plan_stop_instructions) == 1 else clean_itinerary

    monkeypatch.setattr(itinerary, "_plan_stop", fake_plan_stop)

    result = itinerary.edit_graph.invoke(
        make_edit_state(
            macro_plan=macro_plan,
            daily_itinerary=daily_itinerary,
            active_excluded_categories=["food"],
            edit_request="rework Kyoto",
        )
    )

    assert len(plan_stop_instructions) == 2  # initial attempt + one retry
    assert result["validation_feedback"] is None
    assert result["blocking_violation"] is None

    # The retry's instruction actually carried the violation feedback, not
    # just a repeat of the first attempt's instruction.
    assert "Fix these issues" not in plan_stop_instructions[0]
    assert "Fix these issues" in plan_stop_instructions[1]
    assert "Ramen shop" in plan_stop_instructions[1]


def test_edit_graph_blocking_violation_not_committed_after_retry_cap(monkeypatch):
    macro_plan = make_macro_plan()
    daily_itinerary = make_daily_itinerary(macro_plan)

    # The exclusion is already persisted (no delta this turn), so no all-stop
    # expansion fires and the edit stays scoped to Kyoto.
    classification = EditClassification(
        scope="stop",
        target_stops=["Kyoto"],
        note="Rework Kyoto.",
    )

    class FakeClassifyChain:
        def invoke(self, args):
            return classification

    monkeypatch.setattr(itinerary, "edit_classify_chain", FakeClassifyChain())

    always_violating = StopItinerary(
        days=[
            StopDayContent(activities=[_activity("Ramen shop", category="food")]),
            StopDayContent(activities=[_activity("Sushi bar", category="food")]),
        ]
    )
    monkeypatch.setattr(itinerary, "_plan_stop", lambda tr, stop, extra_instruction=None: always_violating)

    result = itinerary.edit_graph.invoke(
        make_edit_state(
            macro_plan=macro_plan,
            daily_itinerary=daily_itinerary,
            active_excluded_categories=["food"],
            edit_request="rework Kyoto",
        )
    )

    # Retries exhausted, violation never fixed.
    assert result["edit_retry_count"] == itinerary.MAX_EDIT_RETRIES
    assert result["blocking_violation"] is not None

    new_macro, new_daily, new_excl, message = _apply_committed_edit(
        macro_plan, daily_itinerary, ["food"], result
    )

    # The still-violating candidate must NOT be committed - baseline preserved,
    # and the previously-active exclusion must not be dropped by a failed turn.
    assert new_macro is macro_plan
    assert new_daily is daily_itinerary
    assert new_excl == ["food"]
    assert message is not None
    assert "safely apply" in message


# --- day-count integrity: _plan_stop_exact, _build_daily_itinerary, validate_edit ---


def test_plan_stop_exact_retries_when_too_many_days(monkeypatch):
    stop = MacroStop(city="Kyoto", country="Japan", days=2, rationale="r")
    trip_request = make_trip_request()

    too_many = StopItinerary(
        days=[
            StopDayContent(activities=[_activity("a")]),
            StopDayContent(activities=[_activity("b")]),
            StopDayContent(activities=[_activity("c")]),
        ]
    )
    correct = StopItinerary(
        days=[
            StopDayContent(activities=[_activity("x")]),
            StopDayContent(activities=[_activity("y")]),
        ]
    )

    calls = []

    def fake_plan_stop(tr, s, extra_instruction=None):
        calls.append(extra_instruction)
        return too_many if len(calls) == 1 else correct

    monkeypatch.setattr(itinerary, "_plan_stop", fake_plan_stop)

    result = _plan_stop_exact(trip_request, stop, extra_instruction="focus on temples")

    assert len(calls) == 2
    assert result is correct
    # The correction is appended to the ORIGINAL instruction, not a replacement.
    assert calls[1].startswith("focus on temples")
    assert "exactly 2 day" in calls[1]
    assert "3 day(s)" in calls[1]  # names what was actually returned


def test_plan_stop_exact_retries_when_too_few_days(monkeypatch):
    stop = MacroStop(city="Kyoto", country="Japan", days=2, rationale="r")
    trip_request = make_trip_request()

    too_few = StopItinerary(days=[StopDayContent(activities=[_activity("a")])])
    correct = StopItinerary(
        days=[
            StopDayContent(activities=[_activity("x")]),
            StopDayContent(activities=[_activity("y")]),
        ]
    )

    calls = []

    def fake_plan_stop(tr, s, extra_instruction=None):
        calls.append(extra_instruction)
        return too_few if len(calls) == 1 else correct

    monkeypatch.setattr(itinerary, "_plan_stop", fake_plan_stop)

    result = _plan_stop_exact(trip_request, stop)

    assert len(calls) == 2
    assert result is correct
    assert "exactly 2 day" in calls[1]
    assert "1 day(s)" in calls[1]


def test_plan_stop_exact_raises_after_retry_cap(monkeypatch):
    stop = MacroStop(city="Kyoto", country="Japan", days=2, rationale="r")
    trip_request = make_trip_request()

    always_wrong = StopItinerary(days=[StopDayContent(activities=[_activity("a")])])
    calls = []

    def fake_plan_stop(tr, s, extra_instruction=None):
        calls.append(extra_instruction)
        return always_wrong

    monkeypatch.setattr(itinerary, "_plan_stop", fake_plan_stop)

    with pytest.raises(ValueError):
        _plan_stop_exact(trip_request, stop)

    # Initial attempt + MAX_STOP_DAY_RETRIES retries, no more.
    assert len(calls) == itinerary.MAX_STOP_DAY_RETRIES + 1


def test_build_daily_itinerary_recovers_on_second_attempt(monkeypatch):
    macro_plan = make_macro_plan()  # Tokyo 3, Kyoto 2, Osaka 2
    trip_request = make_trip_request()

    attempts_by_city = {}

    def fake_plan_stop(tr, stop, extra_instruction=None):
        attempts_by_city[stop.city] = attempts_by_city.get(stop.city, 0) + 1
        if stop.city == "Kyoto" and attempts_by_city[stop.city] == 1:
            return StopItinerary(days=[StopDayContent(activities=[_activity("wrong")])])
        return StopItinerary(
            days=[StopDayContent(activities=[_activity(f"{stop.city} act")]) for _ in range(stop.days)]
        )

    monkeypatch.setattr(itinerary, "_plan_stop", fake_plan_stop)

    result = _build_daily_itinerary(trip_request, macro_plan)

    assert len(result.days) == 7
    assert [d.day_number for d in result.days] == list(range(1, 8))
    assert attempts_by_city["Kyoto"] == 2


def test_build_daily_itinerary_propagates_persistent_day_count_mismatch(monkeypatch):
    macro_plan = make_macro_plan()
    trip_request = make_trip_request()

    def fake_plan_stop(tr, stop, extra_instruction=None):
        if stop.city == "Kyoto":
            return StopItinerary(days=[StopDayContent(activities=[_activity("wrong")])])
        return StopItinerary(
            days=[StopDayContent(activities=[_activity(f"{stop.city} act")]) for _ in range(stop.days)]
        )

    monkeypatch.setattr(itinerary, "_plan_stop", fake_plan_stop)

    with pytest.raises(ValueError):
        _build_daily_itinerary(trip_request, macro_plan)


def test_validate_edit_detects_total_day_count_mismatch():
    macro_plan = make_macro_plan()  # Tokyo 3, Kyoto 2, Osaka 2 = 7
    daily_itinerary = make_daily_itinerary(macro_plan)
    # Append a phantom extra day beyond what the macro plan declares.
    extra_day = DayPlan(day_number=8, city="Osaka", activities=[_activity("phantom")])
    bloated = DailyItinerary(
        days=daily_itinerary.days + [extra_day],
        estimated_total_cost_usd=_total_cost(daily_itinerary.days + [extra_day]),
    )

    state = make_edit_state(
        macro_plan=macro_plan,
        edit_scope="macro",
        candidate_macro_plan=macro_plan,
        candidate_daily_itinerary=bloated,
    )

    result = validate_edit(state)

    assert result["validation_feedback"] is not None
    assert "8 day(s)" in result["validation_feedback"]
    assert "7 day(s)" in result["validation_feedback"]
    assert result["blocking_violation"] is not None


def test_validate_edit_detects_trailing_overflow_with_no_city_mismatch():
    """The case the old city-by-day check alone couldn't catch: an extra day
    appended past the macro plan's last declared day has no expected city to
    compare against (city_by_day.get() returns None there), so only a direct
    count check - not per-day city comparison - catches it."""
    macro_plan = make_macro_plan()  # Tokyo 3, Kyoto 2, Osaka 2 = 7
    daily_itinerary = make_daily_itinerary(macro_plan)
    # Extra day correctly labeled "Osaka" (the last stop) - city label itself
    # is not wrong, only the total count is.
    extra_day = DayPlan(day_number=8, city="Osaka", activities=[_activity("phantom")])
    bloated = DailyItinerary(
        days=daily_itinerary.days + [extra_day],
        estimated_total_cost_usd=_total_cost(daily_itinerary.days + [extra_day]),
    )

    state = make_edit_state(
        macro_plan=macro_plan,
        edit_scope="macro",
        candidate_macro_plan=macro_plan,
        candidate_daily_itinerary=bloated,
    )

    result = validate_edit(state)

    assert result["validation_feedback"] is not None
    assert result["blocking_violation"] is not None
    # No city-mismatch problem should be present - this proves the count
    # check, not the city check, is what caught it.
    assert "is labeled" not in result["validation_feedback"]


def test_validate_edit_checks_candidate_macro_plan_not_baseline():
    """validate_edit must validate candidate_daily_itinerary against
    candidate_macro_plan - never the baseline macro_plan. A macro edit
    legitimately changes the day allocation, so checking against the OLD
    allocation would reject an internally-consistent new plan."""
    baseline_macro = make_macro_plan()  # Tokyo 3, Kyoto 2, Osaka 2 = 7

    # A different, equally valid allocation - same total (7), different split.
    candidate_macro = MacroPlan(
        stops=[
            MacroStop(city="Tokyo", country="Japan", days=2, rationale="r"),
            MacroStop(city="Kyoto", country="Japan", days=3, rationale="r"),
            MacroStop(city="Osaka", country="Japan", days=2, rationale="r"),
        ]
    )

    # Daily itinerary matches the CANDIDATE allocation exactly: days 1-2
    # Tokyo, 3-5 Kyoto, 6-7 Osaka. Against the baseline's allocation (1-3
    # Tokyo, 4-5 Kyoto, 6-7 Osaka), day 3 would wrongly expect "Tokyo" and
    # get "Kyoto" instead - so this test fails if validate_edit mistakenly
    # checks against the baseline.
    candidate_days = []
    day_number = 1
    for stop in candidate_macro.stops:
        for _ in range(stop.days):
            candidate_days.append(
                DayPlan(day_number=day_number, city=stop.city, activities=[_activity("act")])
            )
            day_number += 1
    candidate_daily = DailyItinerary(
        days=candidate_days, estimated_total_cost_usd=_total_cost(candidate_days)
    )

    state = make_edit_state(
        macro_plan=baseline_macro,  # baseline - must NOT be what gets checked
        edit_scope="macro",
        candidate_macro_plan=candidate_macro,
        candidate_daily_itinerary=candidate_daily,
    )

    result = validate_edit(state)

    assert result["validation_feedback"] is None
    assert result["blocking_violation"] is None


def test_apply_stop_edit_rejects_mismatched_splice_length():
    macro_plan = make_macro_plan()
    daily_itinerary = make_daily_itinerary(macro_plan)
    stop = macro_plan.stops[1]  # Kyoto, days 4-5

    with pytest.raises(ValueError):
        _splice_stop_days(
            daily_itinerary.days,
            [4, 5],
            stop,
            [StopDayContent(activities=[_activity("only one day")])],  # 1, not 2
        )


def test_edit_graph_no_phantom_day_survives_for_macro_edit(monkeypatch):
    """End-to-end regression test for the live-reproduced bug: a macro edit
    where one stop's regeneration over-produces days must not result in a
    committed itinerary with a day count exceeding the macro plan's total,
    and must not silently corrupt the baseline either."""
    macro_plan = make_macro_plan()  # Tokyo 3, Kyoto 2, Osaka 2 = 7
    daily_itinerary = make_daily_itinerary(macro_plan)

    classification = EditClassification(
        scope="macro", target_stops=[], note="Give Osaka one more day."
    )

    class FakeClassifyChain:
        def invoke(self, args):
            return classification

    monkeypatch.setattr(itinerary, "edit_classify_chain", FakeClassifyChain())

    # Macro plan correctly reallocates to still sum to 7 - the bug was never
    # in macro_chain's output, it was in the per-stop day-content generation
    # not honoring that allocation.
    reallocated = MacroPlan(
        stops=[
            MacroStop(city="Tokyo", country="Japan", days=2, rationale="r"),
            MacroStop(city="Kyoto", country="Japan", days=2, rationale="r"),
            MacroStop(city="Osaka", country="Japan", days=3, rationale="r"),
        ]
    )
    monkeypatch.setattr(
        itinerary, "macro_chain", type("Fake", (), {"invoke": lambda self, args: reallocated})()
    )

    # Tokyo over-produces (3 days instead of the newly-allocated 2) on every
    # call - simulates the live-reproduced failure mode persisting through
    # _plan_stop_exact's single retry.
    def fake_plan_stop(tr, stop, extra_instruction=None):
        day_count = 3 if stop.city == "Tokyo" else stop.days
        return StopItinerary(
            days=[StopDayContent(activities=[_activity(f"{stop.city} act")]) for _ in range(day_count)]
        )

    monkeypatch.setattr(itinerary, "_plan_stop", fake_plan_stop)

    # apply_macro_edit's _build_daily_itinerary call propagates the raise
    # (via _plan_stop_exact exhausting its retry cap) straight out of
    # edit_graph.invoke - it must never silently produce a phantom-day
    # candidate that then gets committed.
    with pytest.raises(ValueError):
        itinerary.edit_graph.invoke(
            make_edit_state(
                macro_plan=macro_plan,
                daily_itinerary=daily_itinerary,
                edit_request="give Osaka one more day",
            )
        )


# --- Fix 3: durable (but revocable) trip-wide exclusion constraints -----------
#
# Macro plan throughout: Tokyo (3d, days 1-3), Kyoto (2d, days 4-5),
# Osaka (2d, days 6-7) - total 7 days, matching make_trip_request().


def test_durable_exclusion_persists_to_later_unrelated_stop_edit(monkeypatch):
    """Test A: a previously-active exclusion is still proactively injected and
    enforced on a later turn whose text never mentions it, and - since this
    turn carries no delta - the edit stays scoped to the stop actually named."""
    macro_plan = make_macro_plan()
    daily_itinerary = make_daily_itinerary(macro_plan)

    classification = EditClassification(scope="stop", target_stops=["Kyoto"], note="Rework Kyoto.")
    monkeypatch.setattr(itinerary, "edit_classify_chain", type("F", (), {"invoke": lambda self, a: classification})())

    captured_instructions = []

    def fake_plan_stop(trip_request, stop, extra_instruction=None):
        captured_instructions.append(extra_instruction)
        return StopItinerary(
            days=[StopDayContent(activities=[_activity("Temple visit")]) for _ in range(stop.days)]
        )

    monkeypatch.setattr(itinerary, "_plan_stop", fake_plan_stop)

    result = itinerary.edit_graph.invoke(
        make_edit_state(
            macro_plan=macro_plan,
            daily_itinerary=daily_itinerary,
            active_excluded_categories=["food"],
            edit_request="rework Kyoto",
        )
    )

    assert result["effective_excluded_categories"] == ["food"]
    assert result["edit_scope"] == "stop"
    assert result["target_stops"] == ["Kyoto"]  # not expanded - no delta this turn
    assert len(captured_instructions) == 1
    assert "Hard constraint" in captured_instructions[0]
    assert "food" in captured_instructions[0]
    assert result["validation_feedback"] is None
    assert result["blocking_violation"] is None


def test_durable_exclusion_persists_through_macro_edit(monkeypatch):
    """Test B: an unrelated macro (structural) edit still injects the active
    exclusion into every regenerated stop's content instruction, while the
    macro allocation chain itself never sees it (deliberate split)."""
    macro_plan = make_macro_plan()
    reallocated = MacroPlan(
        stops=[
            MacroStop(city="Tokyo", country="Japan", days=2, rationale="r"),
            MacroStop(city="Kyoto", country="Japan", days=2, rationale="r"),
            MacroStop(city="Osaka", country="Japan", days=3, rationale="r"),
        ]
    )

    captured_macro_feedback = []

    class FakeMacroChain:
        def invoke(self, args):
            captured_macro_feedback.append(args["feedback"])
            return reallocated

    monkeypatch.setattr(itinerary, "macro_chain", FakeMacroChain())

    captured_instructions = []

    def fake_plan_stop(trip_request, stop, extra_instruction=None):
        captured_instructions.append(extra_instruction)
        return StopItinerary(
            days=[StopDayContent(activities=[_activity("Temple visit")]) for _ in range(stop.days)]
        )

    monkeypatch.setattr(itinerary, "_plan_stop", fake_plan_stop)

    state = make_edit_state(
        macro_plan=macro_plan,
        edit_request="give Osaka one more day and take it from Tokyo",
        effective_excluded_categories=["food"],
    )
    apply_macro_edit(state)

    assert len(captured_macro_feedback) == 1
    assert "food" not in captured_macro_feedback[0]  # macro allocation never sees it

    assert len(captured_instructions) == 3  # all three stops regenerated
    assert all("food" in instr and "Hard constraint" in instr for instr in captured_instructions)


def test_validation_backstop_blocks_reintroduced_category_without_mention(monkeypatch):
    """Test C: even when the current turn's text never mentions the excluded
    category, validate_edit still blocks its reappearance, and the failed
    commit must leave the previously-active exclusion untouched."""
    macro_plan = make_macro_plan()
    daily_itinerary = make_daily_itinerary(macro_plan)

    classification = EditClassification(scope="stop", target_stops=["Kyoto"], note="Rework Kyoto.")
    monkeypatch.setattr(itinerary, "edit_classify_chain", type("F", (), {"invoke": lambda self, a: classification})())

    always_violating = StopItinerary(
        days=[StopDayContent(activities=[_activity("Ramen shop", category="food")]) for _ in range(2)]
    )
    monkeypatch.setattr(itinerary, "_plan_stop", lambda tr, stop, extra_instruction=None: always_violating)

    result = itinerary.edit_graph.invoke(
        make_edit_state(
            macro_plan=macro_plan,
            daily_itinerary=daily_itinerary,
            active_excluded_categories=["food"],
            edit_request="rework Kyoto",
        )
    )

    assert result["blocking_violation"] is not None

    new_macro, new_daily, new_excl, message = _apply_committed_edit(
        macro_plan, daily_itinerary, ["food"], result
    )
    assert new_macro is macro_plan
    assert new_daily is daily_itinerary
    assert new_excl == ["food"]
    assert message is not None


def test_revocation_allows_category_again_and_persists_empty_set(monkeypatch):
    """Test D: revoking the only active exclusion is a durable change, so it
    forces an all-stop regeneration (never leaves food banned in stops we
    didn't touch) and commits with the exclusion set now empty."""
    macro_plan = make_macro_plan()
    daily_itinerary = make_daily_itinerary(macro_plan)

    classification = EditClassification(
        scope="stop",
        target_stops=["Kyoto"],
        remove_excluded_categories=["food"],
        note="Lift the food exclusion.",
    )
    monkeypatch.setattr(itinerary, "edit_classify_chain", type("F", (), {"invoke": lambda self, a: classification})())

    plan_stop_calls = []

    def fake_plan_stop(trip_request, stop, extra_instruction=None):
        plan_stop_calls.append(stop.city)
        return StopItinerary(
            days=[
                StopDayContent(activities=[_activity(f"{stop.city} food stop", category="food")])
                for _ in range(stop.days)
            ]
        )

    monkeypatch.setattr(itinerary, "_plan_stop", fake_plan_stop)

    result = itinerary.edit_graph.invoke(
        make_edit_state(
            macro_plan=macro_plan,
            daily_itinerary=daily_itinerary,
            active_excluded_categories=["food"],
            edit_request="actually restaurants are fine now",
        )
    )

    assert result["effective_excluded_categories"] == []
    assert sorted(plan_stop_calls) == ["Kyoto", "Osaka", "Tokyo"]  # every stop regenerated
    assert result["validation_feedback"] is None
    assert result["blocking_violation"] is None

    new_macro, new_daily, new_excl, message = _apply_committed_edit(
        macro_plan, daily_itinerary, ["food"], result
    )
    assert new_excl == []
    assert message is None


def test_add_one_exclusion_while_removing_another(monkeypatch):
    """Test E: a single turn can add one durable exclusion and remove another
    at the same time; the merge math nets them independently."""
    classification = EditClassification(
        scope="stop",
        target_stops=["Kyoto"],
        add_excluded_categories=["nightlife"],
        remove_excluded_categories=["food"],
        note="Food is okay now, but no nightlife.",
    )
    monkeypatch.setattr(itinerary, "edit_classify_chain", type("F", (), {"invoke": lambda self, a: classification})())

    state = make_edit_state(active_excluded_categories=["food"], edit_request="switch constraints")
    result = classify_edit(state)

    assert result["effective_excluded_categories"] == ["nightlife"]
    assert result["edit_scope"] == "stop"
    assert result["target_stops"] == ["Tokyo", "Kyoto", "Osaka"]  # durable change -> expanded


def test_failed_turn_does_not_persist_addition(monkeypatch):
    """Test F1: a blocking violation on a turn that was adding a new durable
    exclusion must not let that addition leak into persisted state."""
    macro_plan = make_macro_plan()
    daily_itinerary = make_daily_itinerary(macro_plan)

    classification = EditClassification(
        scope="stop", target_stops=["Kyoto"], add_excluded_categories=["nightlife"], note="No nightlife."
    )
    monkeypatch.setattr(itinerary, "edit_classify_chain", type("F", (), {"invoke": lambda self, a: classification})())

    monkeypatch.setattr(
        itinerary,
        "_plan_stop",
        lambda tr, stop, extra_instruction=None: StopItinerary(
            days=[
                StopDayContent(activities=[_activity("Rooftop bar", category="nightlife")])
                for _ in range(stop.days)
            ]
        ),
    )

    result = itinerary.edit_graph.invoke(
        make_edit_state(
            macro_plan=macro_plan,
            daily_itinerary=daily_itinerary,
            active_excluded_categories=["food"],
            edit_request="no nightlife anywhere",
        )
    )

    assert result["blocking_violation"] is not None

    new_macro, new_daily, new_excl, message = _apply_committed_edit(
        macro_plan, daily_itinerary, ["food"], result
    )
    assert new_excl == ["food"]  # nightlife addition discarded
    assert new_macro is macro_plan


def test_failed_turn_does_not_persist_removal(monkeypatch):
    """Test F2: a blocking violation on a turn that was removing a durable
    exclusion must not let that removal leak into persisted state."""
    macro_plan = make_macro_plan()
    daily_itinerary = make_daily_itinerary(macro_plan)

    classification = EditClassification(
        scope="stop", target_stops=["Kyoto"], remove_excluded_categories=["food"], note="Food is fine now."
    )
    monkeypatch.setattr(itinerary, "edit_classify_chain", type("F", (), {"invoke": lambda self, a: classification})())

    monkeypatch.setattr(
        itinerary,
        "_plan_stop",
        lambda tr, stop, extra_instruction=None: StopItinerary(
            days=[
                StopDayContent(activities=[_activity("Rooftop bar", category="nightlife")])
                for _ in range(stop.days)
            ]
        ),
    )

    result = itinerary.edit_graph.invoke(
        make_edit_state(
            macro_plan=macro_plan,
            daily_itinerary=daily_itinerary,
            active_excluded_categories=["food", "nightlife"],
            edit_request="actually restaurants are fine now",
        )
    )

    assert result["blocking_violation"] is not None  # nightlife still excluded and present

    new_macro, new_daily, new_excl, message = _apply_committed_edit(
        macro_plan, daily_itinerary, ["food", "nightlife"], result
    )
    assert new_excl == ["food", "nightlife"]  # food removal discarded


def test_contradictory_delta_keeps_active_category_active(monkeypatch):
    """Test G1: a classifier that both adds and removes the same category in
    one turn is self-contradictory output - treated as a no-op, so a
    currently-active category stays active (NOT 'remove wins')."""
    classification = EditClassification(
        scope="stop",
        target_stops=["Kyoto"],
        add_excluded_categories=["food"],
        remove_excluded_categories=["food"],
        note="x",
    )
    monkeypatch.setattr(itinerary, "edit_classify_chain", type("F", (), {"invoke": lambda self, a: classification})())

    state = make_edit_state(active_excluded_categories=["food"], edit_request="rework Kyoto")
    result = classify_edit(state)

    assert result["effective_excluded_categories"] == ["food"]
    assert result["edit_scope"] == "stop"
    assert result["target_stops"] == ["Kyoto"]  # no durable change -> no expansion


def test_contradictory_delta_keeps_inactive_category_inactive(monkeypatch):
    """Test G2: the same self-contradiction, but for a category that was not
    already active - it must stay inactive, not get flipped on."""
    classification = EditClassification(
        scope="stop",
        target_stops=["Kyoto"],
        add_excluded_categories=["food"],
        remove_excluded_categories=["food"],
        note="x",
    )
    monkeypatch.setattr(itinerary, "edit_classify_chain", type("F", (), {"invoke": lambda self, a: classification})())

    state = make_edit_state(active_excluded_categories=[], edit_request="rework Kyoto")
    result = classify_edit(state)

    assert result["effective_excluded_categories"] == []
    assert result["target_stops"] == ["Kyoto"]  # no durable change -> no expansion


def test_durable_delta_expands_targets_to_all_stops(monkeypatch):
    """Test H: a classifier that names only one stop but also carries a real
    durable delta is overridden - the code, not the prompt, is what
    guarantees every stop is purged before the exclusion is persisted."""
    classification = EditClassification(
        scope="stop", target_stops=["Kyoto"], add_excluded_categories=["food"], note="No food anywhere."
    )
    monkeypatch.setattr(itinerary, "edit_classify_chain", type("F", (), {"invoke": lambda self, a: classification})())

    state = make_edit_state(active_excluded_categories=[], edit_request="no food anywhere, kosher only")
    result = classify_edit(state)

    assert result["edit_scope"] == "stop"
    assert result["target_stops"] == ["Tokyo", "Kyoto", "Osaka"]
    assert result["effective_excluded_categories"] == ["food"]


def test_unsupported_with_durable_delta_is_promoted_to_all_stop_edit(monkeypatch):
    """Test I: a classifier that misjudges an edit as 'unsupported' but still
    extracted a real durable delta must not have that constraint change
    discarded - it gets promoted to a trip-wide stop edit instead."""
    classification = EditClassification(
        scope="unsupported", target_stops=[], add_excluded_categories=["food"], note="n/a"
    )
    monkeypatch.setattr(itinerary, "edit_classify_chain", type("F", (), {"invoke": lambda self, a: classification})())

    state = make_edit_state(active_excluded_categories=[], edit_request="no food anywhere, kosher only")
    result = classify_edit(state)

    assert result["edit_scope"] == "stop"
    assert result["target_stops"] == ["Tokyo", "Kyoto", "Osaka"]
    assert result["effective_excluded_categories"] == ["food"]


def test_local_scoped_request_stays_local(monkeypatch):
    """Test J: the explicit MVP-boundary regression - a stop-local exclusion
    request ('no food in Kyoto') with no durable delta stays local and is
    NOT treated as a trip-wide constraint, even though it names a category."""
    classification = EditClassification(scope="stop", target_stops=["Kyoto"], note="No food in Kyoto.")
    monkeypatch.setattr(itinerary, "edit_classify_chain", type("F", (), {"invoke": lambda self, a: classification})())

    state = make_edit_state(active_excluded_categories=[], edit_request="no food in Kyoto")
    result = classify_edit(state)

    assert result["edit_scope"] == "stop"
    assert result["target_stops"] == ["Kyoto"]
    assert result["effective_excluded_categories"] == []  # equal to active baseline - no expansion


def test_edit_graph_topology_unchanged():
    """Test K: Fix 3 changes node bodies and the state schema only - the graph
    topology itself (node set) must be unchanged."""
    nodes = set(itinerary.edit_graph.get_graph().nodes.keys())
    assert nodes == {
        "__start__",
        "classify_edit",
        "apply_macro_edit",
        "apply_stop_edit",
        "validate_edit",
        "respond_unsupported",
        "__end__",
    }


def test_classify_edit_merge_math_add_to_empty_active_set(monkeypatch):
    classification = EditClassification(scope="macro", target_stops=[], add_excluded_categories=["food"], note="x")
    monkeypatch.setattr(itinerary, "edit_classify_chain", type("F", (), {"invoke": lambda self, a: classification})())

    state = make_edit_state(active_excluded_categories=[], edit_request="no food anywhere")
    result = classify_edit(state)

    assert result["effective_excluded_categories"] == ["food"]


def test_classify_edit_merge_math_idempotent_restatement(monkeypatch):
    """Re-stating an already-active exclusion changes nothing - effective
    equals active, so no forced expansion fires either."""
    classification = EditClassification(scope="stop", target_stops=["Kyoto"], add_excluded_categories=["food"], note="x")
    monkeypatch.setattr(itinerary, "edit_classify_chain", type("F", (), {"invoke": lambda self, a: classification})())

    state = make_edit_state(active_excluded_categories=["food"], edit_request="still no food, more museums in Kyoto")
    result = classify_edit(state)

    assert result["effective_excluded_categories"] == ["food"]
    assert result["target_stops"] == ["Kyoto"]  # unchanged - not expanded


def test_classify_edit_prompt_receives_active_exclusions(monkeypatch):
    """The {active_exclusions} prompt slot must actually carry the current
    active set - without it the model cannot interpret a revocation."""
    captured = {}

    class FakeClassifyChain:
        def invoke(self, args):
            captured.update(args)
            return EditClassification(scope="stop", target_stops=["Kyoto"], note="x")

    monkeypatch.setattr(itinerary, "edit_classify_chain", FakeClassifyChain())

    classify_edit(make_edit_state(active_excluded_categories=["nightlife", "food"], edit_request="rework Kyoto"))
    assert captured["active_exclusions"] == "food, nightlife"  # sorted, joined

    classify_edit(make_edit_state(active_excluded_categories=[], edit_request="rework Kyoto"))
    assert captured["active_exclusions"] == "none"
