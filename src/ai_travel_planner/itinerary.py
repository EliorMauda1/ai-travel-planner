"""Phase 2+3: LangGraph workflow that turns a TripRequest into a macro + daily
itinerary, grounding daily activities in real Foursquare places search results
via a tool-calling agent sub-flow."""

import os
from typing import Annotated, List, Literal, Optional, TypedDict

from dotenv import load_dotenv
import requests
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from ai_travel_planner.intake import TripRequest, run_intake

load_dotenv()

MAX_MACRO_RETRIES = 2
MAX_TOOL_ROUND_TRIPS = 3
MAX_EDIT_RETRIES = 2
MAX_STOP_DAY_RETRIES = 1

FOURSQUARE_API_KEY = os.getenv("FOURSQUARE_API_KEY")
FOURSQUARE_SEARCH_URL = "https://places-api.foursquare.com/places/search"
FOURSQUARE_API_VERSION = "2025-06-17"

model = ChatOpenAI(model="gpt-4o-mini")


# --- Schemas -----------------------------------------------------------


class MacroStop(BaseModel):
    city: str = Field(description="A concrete city or town to visit.")
    country: Optional[str] = Field(default=None, description="Country the city is in.")
    days: int = Field(description="Number of days to spend at this stop.")
    rationale: str = Field(description="Short reason this stop fits the trip.")


class MacroPlan(BaseModel):
    stops: List[MacroStop] = Field(
        description="Ordered list of stops. days across all stops must sum "
        "exactly to the trip's duration_days."
    )


ActivityCategory = Literal[
    "sightseeing", "food", "museums", "nature", "culture", "shopping", "nightlife", "other"
]
ExcludableCategory = Literal[
    "sightseeing", "food", "museums", "nature", "culture", "shopping", "nightlife"
]


class Activity(BaseModel):
    name: str
    description: str
    category: ActivityCategory = Field(
        description="One of: sightseeing, food, museums, nature, culture, shopping, "
        "nightlife, other."
    )
    estimated_cost_usd: Optional[float] = Field(
        default=None, description="Rough per-activity cost estimate in USD."
    )
    source: Literal["foursquare", "llm_estimate"] = Field(
        description="'foursquare' ONLY if this exact place came back from a "
        "search_places tool result; 'llm_estimate' if it's the model's own "
        "suggestion (search failed, found nothing, or wasn't used)."
    )

# FOR PYTHON
class DayPlan(BaseModel):
    day_number: int
    city: str
    activities: List[Activity]

# FOR PYTHON
class DailyItinerary(BaseModel):
    days: List[DayPlan]
    estimated_total_cost_usd: Optional[float] = None

# FOR AI
class StopDayContent(BaseModel):
    activities: List[Activity]

# FOR AI
class StopItinerary(BaseModel):
    days: List[StopDayContent] = Field(
        description="One entry per day at this stop, in order."
    )


# --- Graph state ---------------------------------------------------------


class ItineraryState(TypedDict):
    trip_request: TripRequest
    macro_plan: Optional[MacroPlan]
    macro_retry_count: int
    validation_feedback: Optional[str]
    daily_itinerary: Optional[DailyItinerary]


# --- Macro plan generation + validation ----------------------------------

macro_extractor = model.with_structured_output(MacroPlan)

MACRO_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a travel planner. Given the trip request below, propose a "
            "macro itinerary: a list of cities/regions to visit and how many "
            "days to spend at each.\n"
            "Rules:\n"
            "- The 'days' across all stops MUST sum to exactly the trip's "
            "duration_days.\n"
            "- If the destination is vague or missing, choose concrete cities "
            "or regions yourself based on the traveler's interests, budget, "
            "and duration. Never ask the user to clarify.\n"
            "- Keep the number of stops reasonable for the trip length so the "
            "pace matches what the traveler asked for.\n"
            "- Give a short rationale for each stop.\n"
            "- Current plan (if editing): {current_plan}. If a current plan is "
            "given, make the minimal change needed to satisfy the traveler's "
            "request — keep the same stops unless the request requires "
            "adding, removing, or reordering them.",
        ),
        ("system", "Trip request: {trip_request}"),
        ("system", "{feedback}"),
    ]
)

macro_chain = MACRO_PROMPT | macro_extractor


def generate_macro_plan(state: ItineraryState) -> dict:
    feedback = state.get("validation_feedback") or "This is the first attempt."
    macro_plan = macro_chain.invoke(
        {
            "trip_request": state["trip_request"].model_dump_json(),
            "feedback": feedback,
            "current_plan": "",
        }
    )
    return {"macro_plan": macro_plan}


def validate_macro_plan(state: ItineraryState) -> dict:
    macro_plan = state["macro_plan"]
    trip_request = state["trip_request"]
    total_days = sum(stop.days for stop in macro_plan.stops)

    if total_days == trip_request.duration_days:
        return {"validation_feedback": None}

    feedback = (
        f"Previous attempt allocated {total_days} total day(s) across stops, "
        f"but the trip is {trip_request.duration_days} day(s) long. Adjust the "
        f"day allocation so the stops sum to exactly {trip_request.duration_days}."
    )
    return {
        "validation_feedback": feedback,
        "macro_retry_count": state["macro_retry_count"] + 1,
    }


def route_after_validation(state: ItineraryState) -> str:
    if state.get("validation_feedback") is None:
        return "valid"
    if state["macro_retry_count"] >= MAX_MACRO_RETRIES:
        return "valid"
    return "retry"


# --- Places search tool (Foursquare) --------------------------------------


def _foursquare_search(city: str, category: str, query: str) -> Optional[List[dict]]:
    """Hit the Foursquare Places API. Returns None on failure, [] on no results."""
    if not FOURSQUARE_API_KEY:
        return None

    search_text = f"{query} {category}".strip() if query else category
    try:
        response = requests.get(
            FOURSQUARE_SEARCH_URL,
            params={"near": city, "query": search_text, "limit": 5},
            headers={
                "Authorization": f"Bearer {FOURSQUARE_API_KEY}",
                "Accept": "application/json",
                "X-Places-Api-Version": FOURSQUARE_API_VERSION,
            },
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException:
        return None

    results = response.json().get("results", [])
    places = []
    for r in results:
        name = r.get("name")
        if not name:
            continue
        categories = r.get("categories") or [{}]
        location = r.get("location") or {}
        places.append(
            {
                "name": name,
                "category": categories[0].get("name"),
                "address": location.get("formatted_address"),
            }
        )
    return places


@tool
def search_places(city: str, category: str, query: str = "") -> str:
    """Search for real, currently-operating restaurants or attractions in a
    city via the Foursquare Places API. `category` should be a short term
    like 'restaurant', 'museum', 'temple', 'park', or 'cafe'. `query` is an
    optional extra keyword to narrow results (e.g. 'ramen', 'anime')."""
    places = _foursquare_search(city, category, query)
    # This means None
    if places is None:
        return (
            "SEARCH_UNAVAILABLE: the places search could not be reached right "
            "now. You may still suggest options from your own knowledge, but "
            "you MUST mark them as unverified (source='llm_estimate'), never "
            "as confirmed/verified."
        )
    # This means []
    if not places:
        return (
            f"SEARCH_NO_RESULTS: no matching places found for "
            f"'{query or category}' in {city}. You may still suggest options "
            f"from your own knowledge, but you MUST mark them as unverified "
            f"(source='llm_estimate')."
        )

    lines = [f"- {p['name']} ({p['category']}) — {p['address']}" for p in places]
    return "SEARCH_RESULTS (verified via Foursquare):\n" + "\n".join(lines)


TOOLS_BY_NAME = {"search_places": search_places}
model_with_tools = model.bind_tools([search_places])


# --- Per-stop tool-calling agent sub-flow ---------------------------------


class StopPlanningState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    tool_round_trips: int
    stop_itinerary: Optional[StopItinerary]


def agent_node(state: StopPlanningState) -> dict:
    ai_message = model_with_tools.invoke(state["messages"])
    return {"messages": [ai_message]}


def tools_node(state: StopPlanningState) -> dict:
    last_message: AIMessage = state["messages"][-1]
    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_fn = TOOLS_BY_NAME[tool_call["name"]]
        result = tool_fn.invoke(tool_call["args"])
        tool_messages.append(
            ToolMessage(content=result, tool_call_id=tool_call["id"])
        )
    return {
        "messages": tool_messages,
        "tool_round_trips": state["tool_round_trips"] + 1,
    }


def route_after_agent(state: StopPlanningState) -> str:
    last_message: AIMessage = state["messages"][-1]
    has_tool_calls = bool(getattr(last_message, "tool_calls", None))
    if has_tool_calls and state["tool_round_trips"] < MAX_TOOL_ROUND_TRIPS:
        return "tools"
    return "finalize"


day_extractor = model.with_structured_output(StopItinerary)

FINALIZE_INSTRUCTION = (
    "Now produce the final structured itinerary for this stop (the exact "
    "number of days requested, 2-4 activities per day). For every activity, "
    "set `source` to 'foursquare' ONLY if it exactly matches a place returned "
    "by a search_places tool result above; otherwise set `source` to "
    "'llm_estimate'. Never mark something 'foursquare' unless it was actually "
    "returned by the tool."
)


def finalize_node(state: StopPlanningState) -> dict:
    messages = state["messages"] + [HumanMessage(content=FINALIZE_INSTRUCTION)]
    stop_itinerary = day_extractor.invoke(messages)
    return {"stop_itinerary": stop_itinerary}


stop_graph = StateGraph(StopPlanningState)
stop_graph.add_node("agent", agent_node)
stop_graph.add_node("tools", tools_node)
stop_graph.add_node("finalize", finalize_node)

stop_graph.set_entry_point("agent")
stop_graph.add_conditional_edges(
    "agent", route_after_agent, {"tools": "tools", "finalize": "finalize"}
)
stop_graph.add_edge("tools", "agent")
stop_graph.add_edge("finalize", END)

stop_planning_graph = stop_graph.compile()


def _plan_stop(
    trip_request: TripRequest,
    stop: MacroStop,
    extra_instruction: Optional[str] = None,
) -> StopItinerary:
    system_message = SystemMessage(
        content=(
            "You are a travel-planning agent building a day-by-day itinerary "
            f"for one stop of a larger trip.\n"
            f"Stop: {stop.city}"
            + (f", {stop.country}" if stop.country else "")
            + f" — {stop.days} day(s). Why this stop: {stop.rationale}\n"
            f"Trip context: {trip_request.model_dump_json()}\n"
            "Use the search_places tool to look up REAL, current restaurants "
            "and attractions matching the traveler's interests before you "
            "finalize the plan. You may call it a few times with different "
            "categories or queries, but stop once you have enough grounded "
            "options — don't over-search.\n"
            "IMPORTANT: only describe a place as verified/confirmed if it came "
            "from a search_places result. If you rely on your own knowledge "
            "instead, say so plainly rather than presenting a guess as "
            "confirmed."
            + (f"\n\n{extra_instruction}" if extra_instruction else "")
        )
    )
    human_message = HumanMessage(
        content=f"Plan {stop.days} day(s) of activities for {stop.city}."
    )

    result = stop_planning_graph.invoke(
        {
            "messages": [system_message, human_message],
            "tool_round_trips": 0,
            "stop_itinerary": None,
        }
    )
    return result["stop_itinerary"]


def _plan_stop_exact(
    trip_request: TripRequest,
    stop: MacroStop,
    extra_instruction: Optional[str] = None,
) -> StopItinerary:
    """Like _plan_stop, but enforces that the result has exactly stop.days
    day entries - retrying (capped at MAX_STOP_DAY_RETRIES) with a corrective
    instruction appended to the ORIGINAL extra_instruction if not, and
    raising rather than silently padding/truncating if it still doesn't
    match. A day-count mismatch here would otherwise flow uncaught into the
    committed itinerary (see the live-reproduced "phantom day" bug)."""
    stop_itinerary = _plan_stop(trip_request, stop, extra_instruction=extra_instruction)
    attempt = 0
    while len(stop_itinerary.days) != stop.days and attempt < MAX_STOP_DAY_RETRIES:
        attempt += 1
        correction = (
            f"Your previous attempt returned {len(stop_itinerary.days)} day(s), "
            f"but this stop is exactly {stop.days} day(s) long. Return exactly "
            f"{stop.days} day entries, no more and no fewer."
        )
        retry_instruction = f"{extra_instruction}\n\n{correction}" if extra_instruction else correction
        stop_itinerary = _plan_stop(trip_request, stop, extra_instruction=retry_instruction)

    if len(stop_itinerary.days) != stop.days:
        raise ValueError(
            f"_plan_stop for '{stop.city}' returned {len(stop_itinerary.days)} "
            f"day(s) after {attempt + 1} attempt(s); expected {stop.days}."
        )
    return stop_itinerary


# --- Daily itinerary generation (drives the per-stop sub-flow) ------------


def _total_cost(days: List[DayPlan]) -> Optional[float]:
    costs = [
        activity.estimated_cost_usd
        for day in days
        for activity in day.activities
        if activity.estimated_cost_usd is not None
    ]
    return sum(costs) if costs else None


def _build_daily_itinerary(
    trip_request: TripRequest,
    macro_plan: MacroPlan,
    extra_instruction: Optional[str] = None,
) -> DailyItinerary:
    days: List[DayPlan] = []
    day_number = 1

    for stop in macro_plan.stops:
        stop_itinerary = _plan_stop_exact(trip_request, stop, extra_instruction=extra_instruction)
        for day_content in stop_itinerary.days:
            days.append(
                DayPlan(
                    day_number=day_number,
                    city=stop.city,
                    activities=day_content.activities,
                )
            )
            day_number += 1

    return DailyItinerary(days=days, estimated_total_cost_usd=_total_cost(days))


def generate_daily_itinerary(state: ItineraryState) -> dict:
    daily_itinerary = _build_daily_itinerary(state["trip_request"], state["macro_plan"])
    return {"daily_itinerary": daily_itinerary}


# --- Graph assembly --------------------------------------------------------

graph = StateGraph(ItineraryState)
graph.add_node("generate_macro_plan", generate_macro_plan)
graph.add_node("validate_macro_plan", validate_macro_plan)
graph.add_node("generate_daily_itinerary", generate_daily_itinerary)

graph.set_entry_point("generate_macro_plan")
graph.add_edge("generate_macro_plan", "validate_macro_plan")
graph.add_conditional_edges(
    "validate_macro_plan",
    route_after_validation,
    {"retry": "generate_macro_plan", "valid": "generate_daily_itinerary"},
)
graph.add_edge("generate_daily_itinerary", END)

itinerary_graph = graph.compile()


def plan_trip(trip_request: TripRequest) -> ItineraryState:
    return itinerary_graph.invoke(
        {
            "trip_request": trip_request,
            "macro_plan": None,
            "macro_retry_count": 0,
            "validation_feedback": None,
            "daily_itinerary": None,
        }
    )


# --- Phase 4: conversational editing of an existing plan -------------------


class EditState(TypedDict):
    trip_request: TripRequest
    macro_plan: MacroPlan
    daily_itinerary: DailyItinerary
    edit_request: str

    # Durable, trip-wide exclusions. active_* is the persisted baseline at the
    # start of this turn and is never mutated by the graph; effective_* is the
    # result after applying this turn's add/remove delta, and is what every
    # downstream node (generation + validation) actually honors.
    active_excluded_categories: List[str]
    effective_excluded_categories: List[str]

    edit_scope: Optional[Literal["macro", "stop", "unsupported"]]
    target_stops: Optional[List[str]]
    classification_note: Optional[str]

    edit_retry_count: int
    validation_feedback: Optional[str]
    blocking_violation: Optional[str]

    candidate_macro_plan: Optional[MacroPlan]
    candidate_daily_itinerary: Optional[DailyItinerary]

    response_message: Optional[str]


class EditClassification(BaseModel):
    scope: Literal["macro", "stop", "unsupported"]
    target_stops: List[str] = Field(
        default_factory=list,
        description="City names (copied verbatim from the current macro plan) "
        "this edit targets. Required and non-empty when scope='stop' - one "
        "city for a single-stop edit, several for a named subset, or every "
        "city for a whole-trip content edit. Empty when scope='macro' or "
        "'unsupported'.",
    )
    add_excluded_categories: List[ExcludableCategory] = Field(
        default_factory=list,
        description="Categories the traveler is newly banning TRIP-WIDE and "
        "durably (e.g. 'I only eat kosher food so no restaurants anywhere', "
        "'we never do nightlife'). Leave EMPTY for a stop-local request like "
        "'no food in Venice' or 'fewer museums in Kyoto' - those are ordinary "
        "content edits, not durable constraints.",
    )
    remove_excluded_categories: List[ExcludableCategory] = Field(
        default_factory=list,
        description="Categories the traveler is explicitly lifting from the "
        "currently-active trip-wide exclusions (e.g. 'actually restaurants are "
        "fine now' -> ['food']). Leave EMPTY unless a previously-stated "
        "constraint is being revoked.",
    )
    note: str = Field(
        description="One sentence. If scope='unsupported', explain why this "
        "can't be applied as a plan edit (e.g. it's a factual question, not a "
        "change request). Otherwise, briefly restate what will change."
    )


edit_classifier = model.with_structured_output(EditClassification)

EDIT_CLASSIFY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You classify a traveler's edit request against their existing "
            "trip plan.\n"
            "Current macro plan (stops in order): {macro_summary}\n"
            "Current daily itinerary (day_number: city): {daily_summary}\n"
            "Currently active trip-wide exclusions: {active_exclusions}\n\n"
            "Decide the scope:\n"
            "- 'macro': the request changes the trip's STRUCTURE - the "
            "allocation of days across stops, or adding/removing/reordering "
            "stops (e.g. 'give Tokyo one more day and take it from Osaka'). "
            "This is about structural change, not how many stops are "
            "affected - a content-only request can touch one stop, several, "
            "or all of them and still belong in 'stop', not 'macro'.\n"
            "- 'stop': the request only changes activities/content within one "
            "or more existing stops, without changing the day allocation "
            "(e.g. 'less museums in Kyoto' -> target_stops=['Kyoto']; "
            "'exclude food in Innsbruck and Cortina, leave Venice alone' -> "
            "target_stops=['Innsbruck', 'Cortina d'Ampezzo']; 'kosher only, "
            "no food anywhere, focus on sightseeing' -> target_stops = every "
            "city in the plan). Set target_stops to the exact city name(s) "
            "(copied verbatim from the macro plan above) this edit touches - "
            "never express this as day numbers.\n"
            "- 'unsupported': the request isn't a plan-edit at all (a factual "
            "question, something unrelated to this trip, or a change this "
            "planner can't make).\n\n"
            "Separately, track DURABLE TRIP-WIDE constraints:\n"
            "- add_excluded_categories: the traveler is newly banning a "
            "category across the WHOLE trip (e.g. 'I only eat kosher food so "
            "no restaurants anywhere', 'we never do nightlife').\n"
            "- remove_excluded_categories: the traveler is lifting one of the "
            "currently active exclusions listed above (e.g. 'actually "
            "restaurants are fine now' -> ['food']).\n"
            "Both are drawn from: sightseeing, food, museums, nature, "
            "culture, shopping, nightlife.\n"
            "IMPORTANT: a stop-local request ('no food in Venice', 'fewer "
            "museums in Kyoto') is NOT a durable constraint - leave BOTH "
            "lists empty and classify it as an ordinary 'stop' edit targeting "
            "only the named stop(s).",
        ),
        ("human", "{edit_request}"),
    ]
)

edit_classify_chain = EDIT_CLASSIFY_PROMPT | edit_classifier


def _validate_stop_targets(macro_plan: MacroPlan, target_stops: List[str]) -> Optional[str]:
    """Returns an error string if target_stops is unusable for a stop-level
    edit, or None if every name resolves to an existing stop."""
    if not target_stops:
        return "no target stop(s) were identified for this edit."

    known_cities = {stop.city.lower() for stop in macro_plan.stops}
    unknown = [name for name in target_stops if name.lower() not in known_cities]
    if unknown:
        return f"stop(s) {unknown} don't match any stop in the current plan."
    return None


def classify_edit(state: EditState) -> dict:
    macro_plan = state["macro_plan"]
    daily_itinerary = state["daily_itinerary"]
    active = set(state["active_excluded_categories"])

    macro_summary = "; ".join(f"{stop.city} ({stop.days}d)" for stop in macro_plan.stops)
    daily_summary = "; ".join(
        f"day {day.day_number}: {day.city}" for day in daily_itinerary.days
    )

    classification = edit_classify_chain.invoke(
        {
            "macro_summary": macro_summary,
            "daily_summary": daily_summary,
            "active_exclusions": ", ".join(sorted(active)) or "none",
            "edit_request": state["edit_request"],
        }
    )

    scope = classification.scope
    target_stops = classification.target_stops or None
    note = classification.note

    # Durable trip-wide exclusion delta. A category appearing in BOTH lists is
    # contradictory classifier output and is treated as a no-op for that
    # category, leaving its previous active state untouched.
    add = set(classification.add_excluded_categories)
    remove = set(classification.remove_excluded_categories)
    conflicting = add & remove
    effective_set = (active | (add - conflicting)) - (remove - conflicting)
    effective = sorted(effective_set)

    # If the durable set actually changed, the whole trip must be regenerated
    # under the new constraints - otherwise we would persist a trip-wide
    # exclusion while leaving prohibited content in stops we never touched.
    # (When effective_set == active there is nothing new to purge: the turn
    # that introduced the constraint already regenerated every stop.)
    if effective_set != active and scope != "macro":
        # Covers a partially-targeted 'stop' edit AND an 'unsupported'
        # classification that nonetheless carried a real constraint change.
        scope = "stop"
        target_stops = [stop.city for stop in macro_plan.stops]
        note = (
            "Applying trip-wide exclusions "
            f"({', '.join(effective) or 'none'}) across every stop."
        )
    # scope == "macro" needs no expansion: macro regeneration already rebuilds
    # every stop's day content.

    if scope == "stop":
        error = _validate_stop_targets(macro_plan, target_stops or [])
        if error:
            scope = "unsupported"
            note = f"Could not resolve the target stop(s) for this edit: {error}"

    return {
        "edit_scope": scope,
        "target_stops": target_stops,
        "classification_note": note,
        "effective_excluded_categories": effective,
    }


def _resolve_stops(macro_plan: MacroPlan, target_stops: List[str]) -> List[MacroStop]:
    """The stops named in target_stops, in trip order. Assumes target_stops
    has already been validated (_validate_stop_targets) to only contain
    known city names."""
    wanted = {name.lower() for name in target_stops}
    return [stop for stop in macro_plan.stops if stop.city.lower() in wanted]


def _exclusion_note(effective_excluded_categories: List[str]) -> str:
    """The generation-instruction fragment that proactively tells the planner
    which categories are banned. Shared by both apply nodes so they emit
    identical wording; validate_edit stays a backstop, not the teacher."""
    if not effective_excluded_categories:
        return ""
    return (
        f" Hard constraint: do NOT include any activities in these "
        f"categories: {', '.join(effective_excluded_categories)}."
    )


def _splice_stop_days(
    baseline_days: List[DayPlan],
    target_day_numbers: List[int],
    stop: MacroStop,
    new_days: List[StopDayContent],
) -> List[DayPlan]:
    if len(new_days) != len(target_day_numbers):
        # zip() would otherwise silently truncate to the shorter list,
        # leaving some target days holding stale baseline content.
        raise ValueError(
            f"_splice_stop_days got {len(new_days)} new day(s) for "
            f"{len(target_day_numbers)} target day number(s) in '{stop.city}'; "
            "these must match exactly."
        )
    replacement = {
        day_number: DayPlan(day_number=day_number, city=stop.city, activities=content.activities)
        for day_number, content in zip(sorted(target_day_numbers), new_days)
    }
    return [replacement.get(day.day_number, day) for day in baseline_days]


def apply_macro_edit(state: EditState) -> dict:
    trip_request = state["trip_request"]
    macro_plan = state["macro_plan"]
    feedback = state.get("validation_feedback")

    current_plan = "; ".join(f"{stop.city} ({stop.days}d)" for stop in macro_plan.stops)

    macro_instruction = f"Traveler wants this change: {state['edit_request']}."
    if feedback:
        macro_instruction += f" {feedback}"

    # Macro allocation is about cities and day counts; activity-category
    # exclusions belong only in the day-content regeneration below.
    candidate_macro_plan = macro_chain.invoke(
        {
            "trip_request": trip_request.model_dump_json(),
            "feedback": macro_instruction,
            "current_plan": current_plan,
        }
    )
    content_instruction = macro_instruction + _exclusion_note(
        state["effective_excluded_categories"]
    )
    candidate_daily_itinerary = _build_daily_itinerary(
        trip_request, candidate_macro_plan, extra_instruction=content_instruction
    )
    return {
        "candidate_macro_plan": candidate_macro_plan,
        "candidate_daily_itinerary": candidate_daily_itinerary,
    }


def _stop_day_numbers(macro_plan: MacroPlan, stop: MacroStop) -> List[int]:
    """The full, contiguous range of day_numbers a stop owns in the macro plan."""
    day_number = 1
    for candidate in macro_plan.stops:
        if candidate is stop:
            return list(range(day_number, day_number + stop.days))
        day_number += candidate.days
    raise ValueError(f"stop '{stop.city}' not found in macro_plan.")


def apply_stop_edit(state: EditState) -> dict:
    trip_request = state["trip_request"]
    macro_plan = state["macro_plan"]
    baseline_days = state["daily_itinerary"].days
    target_stops = state["target_stops"]
    feedback = state.get("validation_feedback")

    # classify_edit has already validated target_stops only contains known
    # city names; this resolution is expected to succeed.
    stops = _resolve_stops(macro_plan, target_stops)

    exclusion_note = _exclusion_note(state["effective_excluded_categories"])

    new_days = baseline_days
    for stop in stops:
        stop_day_numbers = _stop_day_numbers(macro_plan, stop)
        previous_activities = [
            activity.name
            for day in baseline_days
            if day.day_number in stop_day_numbers
            for activity in day.activities
        ]
        instruction = (
            f"Traveler edit request: '{state['edit_request']}'. "
            f"Previously planned activities for this stop: "
            f"{', '.join(previous_activities) or 'none'}."
            f"{exclusion_note}"
        )
        if feedback:
            instruction += f" {feedback}"

        stop_itinerary = _plan_stop_exact(trip_request, stop, extra_instruction=instruction)
        new_days = _splice_stop_days(new_days, stop_day_numbers, stop, stop_itinerary.days)

    candidate_daily_itinerary = DailyItinerary(
        days=new_days, estimated_total_cost_usd=_total_cost(new_days)
    )
    return {
        "candidate_macro_plan": macro_plan,
        "candidate_daily_itinerary": candidate_daily_itinerary,
    }


def validate_edit(state: EditState) -> dict:
    macro_plan = state["candidate_macro_plan"]
    daily_itinerary = state["candidate_daily_itinerary"]
    trip_request = state["trip_request"]

    problems = []
    blocking_problems = []

    total_days = sum(stop.days for stop in macro_plan.stops)
    if total_days != trip_request.duration_days:
        problems.append(f"stop days sum to {total_days}, expected {trip_request.duration_days}")

    if len(daily_itinerary.days) != total_days:
        # A raw count mismatch against the (candidate) macro plan - not just
        # self-referential contiguity - catches overflow/underflow even when
        # it lands entirely past the macro plan's last declared day (where
        # city_by_day below has nothing to compare against and would
        # otherwise miss it silently). Self-contradictory, not best-effort
        # safe: subsequent edits derive day ranges from macro_plan, so a
        # surviving mismatch would corrupt every future edit's targeting.
        day_count_problem = (
            f"daily itinerary has {len(daily_itinerary.days)} day(s) but the "
            f"macro plan's stops sum to {total_days} day(s)"
        )
        problems.append(day_count_problem)
        blocking_problems.append(day_count_problem)

    day_numbers = [day.day_number for day in daily_itinerary.days]
    expected_numbers = list(range(1, len(day_numbers) + 1))
    if sorted(day_numbers) != expected_numbers:
        problems.append(f"day numbers {sorted(day_numbers)} are not contiguous from 1")

    city_by_day = {}
    day_number = 1
    for stop in macro_plan.stops:
        for _ in range(stop.days):
            city_by_day[day_number] = stop.city
            day_number += 1
    for day in daily_itinerary.days:
        expected_city = city_by_day.get(day.day_number)
        if expected_city is not None and day.city != expected_city:
            problems.append(
                f"day {day.day_number} is labeled '{day.city}' but the macro "
                f"plan expects '{expected_city}'"
            )

    # The EFFECTIVE set, so a persisted exclusion is still enforced on later
    # turns whose text never mentions it.
    excluded_categories = state["effective_excluded_categories"]
    if excluded_categories:
        if state["edit_scope"] == "stop":
            touched_stops = _resolve_stops(macro_plan, state["target_stops"])
            touched_days = {
                day_number
                for stop in touched_stops
                for day_number in _stop_day_numbers(macro_plan, stop)
            }
        else:
            touched_days = {day.day_number for day in daily_itinerary.days}

        excluded_set = set(excluded_categories)
        for day in daily_itinerary.days:
            if day.day_number not in touched_days:
                continue
            for activity in day.activities:
                if activity.category in excluded_set:
                    violation = (
                        f"day {day.day_number} contains a '{activity.category}' "
                        f"activity ({activity.name}) despite the exclusion request"
                    )
                    problems.append(violation)
                    blocking_problems.append(violation)

    if not problems:
        return {"validation_feedback": None, "blocking_violation": None}

    return {
        "validation_feedback": "Fix these issues: " + "; ".join(problems),
        "edit_retry_count": state["edit_retry_count"] + 1,
        "blocking_violation": "; ".join(blocking_problems) if blocking_problems else None,
    }


def respond_unsupported(state: EditState) -> dict:
    return {
        "response_message": f"I can't apply that as a plan edit: {state['classification_note']}"
    }


def route_after_classify(state: EditState) -> str:
    return state["edit_scope"]


def route_after_edit_validation(state: EditState) -> str:
    if state["validation_feedback"] is None:
        return "done"
    if state["edit_retry_count"] >= MAX_EDIT_RETRIES:
        return "done"
    return "retry_macro" if state["edit_scope"] == "macro" else "retry_stop"


edit_graph_builder = StateGraph(EditState)
edit_graph_builder.add_node("classify_edit", classify_edit)
edit_graph_builder.add_node("apply_macro_edit", apply_macro_edit)
edit_graph_builder.add_node("apply_stop_edit", apply_stop_edit)
edit_graph_builder.add_node("validate_edit", validate_edit)
edit_graph_builder.add_node("respond_unsupported", respond_unsupported)

edit_graph_builder.set_entry_point("classify_edit")
edit_graph_builder.add_conditional_edges(
    "classify_edit",
    route_after_classify,
    {
        "macro": "apply_macro_edit",
        "stop": "apply_stop_edit",
        "unsupported": "respond_unsupported",
    },
)
edit_graph_builder.add_edge("apply_macro_edit", "validate_edit")
edit_graph_builder.add_edge("apply_stop_edit", "validate_edit")
edit_graph_builder.add_conditional_edges(
    "validate_edit",
    route_after_edit_validation,
    {
        "retry_macro": "apply_macro_edit",
        "retry_stop": "apply_stop_edit",
        "done": END,
    },
)
edit_graph_builder.add_edge("respond_unsupported", END)

edit_graph = edit_graph_builder.compile()


def _apply_committed_edit(
    macro_plan: MacroPlan,
    daily_itinerary: DailyItinerary,
    active_excluded_categories: List[str],
    result: dict,
):
    """Single source of truth for whether a turn commits: the candidate plan
    and the candidate exclusions are adopted together, or neither is. Callers
    reassign all four returned values unconditionally, so persistence is never
    inferred from message formatting or any other indirect signal."""
    if result["edit_scope"] == "unsupported":
        return (
            macro_plan,
            daily_itinerary,
            active_excluded_categories,
            result["response_message"],
        )

    blocking_violation = result.get("blocking_violation")
    if blocking_violation:
        return (
            macro_plan,
            daily_itinerary,
            active_excluded_categories,
            f"I couldn't safely apply that edit: {blocking_violation}. "
            "Your itinerary wasn't changed — try rephrasing the request.",
        )

    return (
        result["candidate_macro_plan"],
        result["candidate_daily_itinerary"],
        result["effective_excluded_categories"],
        None,
    )


def _print_result(result: dict) -> None:
    macro_plan: MacroPlan = result["macro_plan"]
    daily_itinerary: DailyItinerary = result["daily_itinerary"]

    print("\nMacro plan:")
    for stop in macro_plan.stops:
        location = f"{stop.city}, {stop.country}" if stop.country else stop.city
        print(f"  {location} — {stop.days} day(s) — {stop.rationale}")

    print("\nDaily itinerary:")
    for day in daily_itinerary.days:
        print(f"\nDay {day.day_number} — {day.city}")
        for activity in day.activities:
            cost = (
                f" (${activity.estimated_cost_usd:.0f})"
                if activity.estimated_cost_usd is not None
                else ""
            )
            tag = (
                "[Foursquare-verified]"
                if activity.source == "foursquare"
                else "[LLM estimate - unverified]"
            )
            print(
                f"  - {tag} [{activity.category}] {activity.name}{cost}: "
                f"{activity.description}"
            )

    if daily_itinerary.estimated_total_cost_usd is not None:
        print(f"\nEstimated total cost: ${daily_itinerary.estimated_total_cost_usd:.0f}")


if __name__ == "__main__":
    trip = run_intake()
    print("\nGenerating itinerary...")
    result = plan_trip(trip)
    macro_plan = result["macro_plan"]
    daily_itinerary = result["daily_itinerary"]
    _print_result(result)

    active_excluded_categories: List[str] = []

    print("\nYou can now ask for changes (or type 'done' to finish).")
    while True:
        message = input("> ").strip()
        if not message or message.lower() == "done":
            break

        edit_result = edit_graph.invoke(
            {
                "trip_request": trip,
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
        )
        (
            macro_plan,
            daily_itinerary,
            active_excluded_categories,
            message_out,
        ) = _apply_committed_edit(
            macro_plan, daily_itinerary, active_excluded_categories, edit_result
        )
        if message_out:
            print(message_out)
        else:
            print("\nUpdated.")
            _print_result({"macro_plan": macro_plan, "daily_itinerary": daily_itinerary})
