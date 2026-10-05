"""Phase 6: direct tests for intake.py's own logic (missing_essentials),
which before this file was only exercised indirectly via ui_state.intake_turn
stubs. missing_essentials gates every user's first interaction with the
product, so it's tested directly here.
"""

from ai_travel_planner.intake import TripRequest, missing_essentials


def test_missing_essentials_returns_all_when_empty():
    missing = missing_essentials(TripRequest())
    assert len(missing) == 3
    assert "trip duration in days" in missing
    assert "total budget in US dollars" in missing
    assert "number of travelers" in missing


def test_missing_essentials_returns_none_when_all_set():
    trip = TripRequest(duration_days=7, budget_usd=2000, traveler_count=2)
    assert missing_essentials(trip) == []


def test_missing_essentials_returns_only_unset_essential_fields():
    trip = TripRequest(duration_days=7, budget_usd=None, traveler_count=2)
    missing = missing_essentials(trip)
    assert missing == ["total budget in US dollars"]


def test_missing_essentials_treats_nonpositive_duration_as_missing():
    trip = TripRequest(duration_days=0, budget_usd=2000, traveler_count=2)
    assert "trip duration in days" in missing_essentials(trip)

    trip = TripRequest(duration_days=-3, budget_usd=2000, traveler_count=2)
    assert "trip duration in days" in missing_essentials(trip)


def test_missing_essentials_treats_nonpositive_budget_as_missing():
    trip = TripRequest(duration_days=7, budget_usd=0, traveler_count=2)
    assert "total budget in US dollars" in missing_essentials(trip)

    trip = TripRequest(duration_days=7, budget_usd=-100, traveler_count=2)
    assert "total budget in US dollars" in missing_essentials(trip)


def test_missing_essentials_treats_nonpositive_traveler_count_as_missing():
    trip = TripRequest(duration_days=7, budget_usd=2000, traveler_count=0)
    assert "number of travelers" in missing_essentials(trip)

    trip = TripRequest(duration_days=7, budget_usd=2000, traveler_count=-1)
    assert "number of travelers" in missing_essentials(trip)


def test_missing_essentials_accepts_valid_positive_values():
    trip = TripRequest(duration_days=1, budget_usd=1, traveler_count=1)
    assert missing_essentials(trip) == []
