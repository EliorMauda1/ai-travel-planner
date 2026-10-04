"""Phase 5: UI-framework-agnostic orchestration glue between the Streamlit app
(ui.py) and the existing Phase 1-4 backend (intake.py, itinerary.py).

Contains no Streamlit imports so it can be unit-tested directly. Every call
into the backend goes through the `intake`/`itinerary` module objects (not
`from ... import name`) so tests can monkeypatch chains/graphs the same way
tests/test_editing.py already does.
"""

import logging
from typing import List, Optional, Tuple

from ai_travel_planner import intake, itinerary
from ai_travel_planner.intake import TripRequest
from ai_travel_planner.itinerary import DailyItinerary, MacroPlan

logger = logging.getLogger(__name__)


def intake_turn(trip_request: TripRequest, message: str) -> Tuple[TripRequest, Optional[str], bool]:
    """Merge one user message into trip_request.

    Returns (trip_request, message, is_error):
    - Follow-up needed:  (updated_request, follow_up_question, False)
    - Intake complete:   (updated_request, None,               False)
    - Operation failed:  (trip_request (unchanged), error_message, True)

    The `is_error` flag is what lets the caller distinguish a normal
    clarification question from an operational failure - both are non-None
    strings, but only one of them should be treated as "still collecting
    intake" versus "something broke, state didn't advance."
    """
    try:
        updated = intake.extraction_chain.invoke(
            {"known_state": trip_request.model_dump_json(), "message": message}
        )

        missing = intake.missing_essentials(updated)
        if not missing:
            return updated, None, False

        question = intake.followup_chain.invoke({"missing_fields": ", ".join(missing)})
        return updated, question.content, False
    except Exception:
        logger.exception("intake_turn failed")
        return (
            trip_request,
            "Something went wrong while processing that. Please try again.",
            True,
        )


def build_initial_edit_state(
    trip_request: TripRequest,
    macro_plan: MacroPlan,
    daily_itinerary: DailyItinerary,
    active_excluded_categories: List[str],
    message: str,
) -> dict:
    """The exact 15-key EditState shape the CLI __main__ loop already builds
    inline, centralized here so it's defined once. The active exclusions are
    copied, not aliased, so later session-state mutation can't reach into a
    live graph run."""
    return {
        "trip_request": trip_request,
        "macro_plan": macro_plan,
        "daily_itinerary": daily_itinerary,
        "edit_request": message,
        "active_excluded_categories": list(active_excluded_categories),
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


def generate_initial_plan(
    trip_request: TripRequest,
) -> Tuple[Optional[MacroPlan], Optional[DailyItinerary], Optional[str]]:
    """Wraps itinerary.plan_trip().

    Returns (macro_plan, daily_itinerary, None) on success, or
    (None, None, error_message) on failure.
    """
    try:
        result = itinerary.plan_trip(trip_request)
        return result["macro_plan"], result["daily_itinerary"], None
    except Exception:
        logger.exception("generate_initial_plan failed")
        return (
            None,
            None,
            "Something went wrong while generating your itinerary. Please try again.",
        )


def run_edit_turn(
    trip_request: TripRequest,
    macro_plan: MacroPlan,
    daily_itinerary: DailyItinerary,
    active_excluded_categories: List[str],
    message: str,
) -> Tuple[MacroPlan, DailyItinerary, List[str], Optional[str]]:
    """Wraps itinerary.edit_graph.invoke() + itinerary._apply_committed_edit().

    On success (supported edit, unsupported request, or a best-effort commit
    after the retry cap): returns the committed plan plus the active trip-wide
    exclusions after this turn, and an optional response_message. On an
    unexpected exception: returns the unchanged baseline plan AND the unchanged
    exclusions plus a friendly error message, never raises - a failed turn must
    not leak constraint additions or removals into persistent state.
    """
    try:
        initial_state = build_initial_edit_state(
            trip_request, macro_plan, daily_itinerary, active_excluded_categories, message
        )
        result = itinerary.edit_graph.invoke(initial_state)
        return itinerary._apply_committed_edit(
            macro_plan, daily_itinerary, active_excluded_categories, result
        )
    except Exception:
        logger.exception("run_edit_turn failed")
        return (
            macro_plan,
            daily_itinerary,
            active_excluded_categories,
            "Something went wrong while updating your plan. Please try again.",
        )
