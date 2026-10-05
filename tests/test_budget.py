"""Phase 6: tests for budget-correctness helpers (_uncosted_activity_count,
_budget_warning). _budget_warning must only ever warn about overage - it
must never claim a trip is "within budget", since known activity costs are
a partial figure that excludes flights/lodging/transport.
"""

from ai_travel_planner.intake import TripRequest
from ai_travel_planner.itinerary import (
    Activity,
    DailyItinerary,
    DayPlan,
    _budget_warning,
    _uncosted_activity_count,
)


def _activity(cost):
    return Activity(
        name="x", description="d", category="sightseeing", estimated_cost_usd=cost, source="llm_estimate"
    )


def _days(costs):
    return [DayPlan(day_number=i + 1, city="Tokyo", activities=[_activity(c)]) for i, c in enumerate(costs)]


def _trip(budget_usd):
    return TripRequest(destination="Japan", duration_days=1, budget_usd=budget_usd, traveler_count=1)


def _itinerary(cost):
    return DailyItinerary(days=[], estimated_total_cost_usd=cost)


# --- _uncosted_activity_count -------------------------------------------------


def test_uncosted_activity_count_zero_when_all_costed():
    assert _uncosted_activity_count(_days([10.0, 20.0])) == 0


def test_uncosted_activity_count_counts_only_missing_estimates():
    assert _uncosted_activity_count(_days([10.0, None, None])) == 2


# --- _budget_warning -----------------------------------------------------------


def test_budget_warning_returns_none_when_budget_unset():
    trip = _trip(budget_usd=None)
    assert _budget_warning(trip, _itinerary(500.0)) is None


def test_budget_warning_returns_none_when_cost_unset():
    trip = _trip(budget_usd=100.0)
    assert _budget_warning(trip, _itinerary(None)) is None


def test_budget_warning_returns_none_when_cost_equals_budget():
    # Boundary: exactly at budget must not be treated as "over".
    trip = _trip(budget_usd=100.0)
    assert _budget_warning(trip, _itinerary(100.0)) is None


def test_budget_warning_returns_none_when_cost_under_budget():
    # Must never claim "within budget" - the only correct output for an
    # under-budget activity cost is silence, since flights/lodging/transport
    # aren't estimated and could still push the real trip over budget.
    trip = _trip(budget_usd=1000.0)
    warning = _budget_warning(trip, _itinerary(200.0))
    assert warning is None


def test_budget_warning_reports_overage_with_disclaimer():
    trip = _trip(budget_usd=1000.0)
    warning = _budget_warning(trip, _itinerary(1250.0))
    assert warning is not None
    assert "$250" in warning
    assert "$1000" in warning
    assert "Flights, lodging, and transport are not included" in warning
