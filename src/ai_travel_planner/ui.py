"""Phase 5: Streamlit UI for the AI Travel Planner.

Reuses the Phase 1-4 backend unmodified via ui_state.py. This file owns only
Streamlit-specific concerns: session lifecycle, chat rendering, and
itinerary rendering.
"""

import os

from dotenv import load_dotenv
import streamlit as st

st.set_page_config(page_title="AI Travel Planner", layout="wide")

# Preflight config check MUST happen before importing the backend - itinerary.py
# constructs a ChatOpenAI client at module import time, so checking after that
# import would already be too late (the import itself would have raised).
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not set. Add it to your .env file and restart the app.")
    st.stop()

from ai_travel_planner import ui_state
from ai_travel_planner.intake import TripRequest
from ai_travel_planner.itinerary import (
    DailyItinerary,
    MacroPlan,
    _budget_warning,
    _uncosted_activity_count,
)

st.title("AI Travel Planner")


def _init_session_state() -> None:
    if "stage" not in st.session_state:
        st.session_state.stage = "collecting_intake"
        st.session_state.trip_request = TripRequest()
        st.session_state.macro_plan = None
        st.session_state.daily_itinerary = None
        # Durable trip-wide exclusions, carried across edit turns.
        st.session_state.active_excluded_categories = []
        st.session_state.chat_history = [
            ("assistant", "Tell me about the trip you want to plan.")
        ]


def _append(role: str, content: str) -> None:
    st.session_state.chat_history.append((role, content))


def render_itinerary(
    trip_request: TripRequest, macro_plan: MacroPlan, daily_itinerary: DailyItinerary
) -> None:
    st.subheader("Macro plan")
    for stop in macro_plan.stops:
        location = f"{stop.city}, {stop.country}" if stop.country else stop.city
        st.markdown(f"**{location}** — {stop.days} day(s)  \n{stop.rationale}")

    st.subheader("Daily itinerary")
    for day in daily_itinerary.days:
        st.markdown(f"**Day {day.day_number} — {day.city}**")
        for activity in day.activities:
            tag = "\U0001F7E2 Foursquare-verified" if activity.source == "foursquare" else "\U0001F7E1 LLM estimate"
            cost = (
                f" (${activity.estimated_cost_usd:.0f})"
                if activity.estimated_cost_usd is not None
                else ""
            )
            st.markdown(
                f"- {tag} · **[{activity.category}] {activity.name}**{cost}: {activity.description}"
            )

    if daily_itinerary.estimated_total_cost_usd is not None:
        uncosted = _uncosted_activity_count(daily_itinerary.days)
        caveat = f" ({uncosted} activity(ies) without a cost estimate)" if uncosted else ""
        st.markdown(
            f"**Estimated activity cost:** ${daily_itinerary.estimated_total_cost_usd:.0f}{caveat}"
        )

    budget_warning = _budget_warning(trip_request, daily_itinerary)
    if budget_warning:
        # st.warning renders markdown, and a pair of "$" is interpreted as
        # inline LaTeX math - the warning text has two ($overage, $budget),
        # so they must be escaped or the dollar amounts silently vanish.
        st.warning(budget_warning.replace("$", "\\$"))


def _handle_message(message: str) -> None:
    """The single call site for every ui_state entry point. Dispatches on
    st.session_state.stage per the approved lifecycle model - see the Phase 5
    plan, section 5a, for the full transition table and the anti-corruption
    guarantee this function is the implementation of."""
    _append("user", message)
    stage = st.session_state.stage

    if stage == "collecting_intake":
        with st.spinner("Thinking..."):
            trip_request, reply, is_error = ui_state.intake_turn(st.session_state.trip_request, message)
        st.session_state.trip_request = trip_request

        if is_error:
            _append("assistant", reply)
            return

        if reply is not None:
            _append("assistant", reply)
            return

        with st.spinner("Planning your trip..."):
            macro_plan, daily_itinerary, error = ui_state.generate_initial_plan(trip_request)

        if error:
            st.session_state.stage = "plan_failed"
            _append(
                "assistant",
                f"{error} Your trip details are saved - send \"retry\" to try planning again.",
            )
        else:
            st.session_state.stage = "planned"
            st.session_state.macro_plan = macro_plan
            st.session_state.daily_itinerary = daily_itinerary
            _append("assistant", "Here's your itinerary! Ask me for any changes.")

    elif stage == "plan_failed":
        if message.strip().lower() != "retry":
            _append(
                "assistant",
                "Planning failed earlier and your trip details are still saved. "
                "Send \"retry\" to try planning again.",
            )
            return

        with st.spinner("Planning your trip..."):
            macro_plan, daily_itinerary, error = ui_state.generate_initial_plan(
                st.session_state.trip_request
            )

        if error:
            _append(
                "assistant",
                f"{error} Your trip details are still saved - send \"retry\" to try again.",
            )
        else:
            st.session_state.stage = "planned"
            st.session_state.macro_plan = macro_plan
            st.session_state.daily_itinerary = daily_itinerary
            _append("assistant", "Here's your itinerary! Ask me for any changes.")

    elif stage == "planned":
        with st.spinner("Applying your edit..."):
            (
                macro_plan,
                daily_itinerary,
                active_excluded_categories,
                response_message,
            ) = ui_state.run_edit_turn(
                st.session_state.trip_request,
                st.session_state.macro_plan,
                st.session_state.daily_itinerary,
                st.session_state.active_excluded_categories,
                message,
            )
        # run_edit_turn returns the previous exclusions unchanged on any
        # non-committing outcome, so this assignment is safe unconditionally.
        st.session_state.macro_plan = macro_plan
        st.session_state.daily_itinerary = daily_itinerary
        st.session_state.active_excluded_categories = active_excluded_categories
        _append("assistant", response_message or "Updated your itinerary.")


_init_session_state()

chat_col, itinerary_col = st.columns(2)

with chat_col:
    st.subheader("Chat")
    for role, content in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(content)

    if message := st.chat_input("Tell me about your trip, or ask for a change..."):
        _handle_message(message)
        st.rerun()

with itinerary_col:
    st.subheader("Itinerary")
    if st.session_state.macro_plan and st.session_state.daily_itinerary:
        render_itinerary(
            st.session_state.trip_request, st.session_state.macro_plan, st.session_state.daily_itinerary
        )
    else:
        st.info("Your itinerary will appear here once planning is complete.")
