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


class Activity(BaseModel):
    name: str
    description: str
    category: str = Field(description="e.g. sightseeing, food, nature, culture")
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

    if places is None:
        return (
            "SEARCH_UNAVAILABLE: the places search could not be reached right "
            "now. You may still suggest options from your own knowledge, but "
            "you MUST mark them as unverified (source='llm_estimate'), never "
            "as confirmed/verified."
        )
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


# --- Daily itinerary generation (drives the per-stop sub-flow) ------------


def _total_cost(days: List[DayPlan]) -> Optional[float]:
    costs = [
        activity.estimated_cost_usd
        for day in days
        for activity in day.activities
        if activity.estimated_cost_usd is not None
    ]
    return sum(costs) if costs else None


def _build_daily_itinerary(trip_request: TripRequest, macro_plan: MacroPlan) -> DailyItinerary:
    days: List[DayPlan] = []
    day_number = 1

    for stop in macro_plan.stops:
        stop_itinerary = _plan_stop(trip_request, stop)
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

    edit_scope: Optional[Literal["macro", "stop", "unsupported"]]
    target_day_numbers: Optional[List[int]]
    classification_note: Optional[str]

    edit_retry_count: int
    validation_feedback: Optional[str]

    candidate_macro_plan: Optional[MacroPlan]
    candidate_daily_itinerary: Optional[DailyItinerary]

    response_message: Optional[str]


class EditClassification(BaseModel):
    scope: Literal["macro", "stop", "unsupported"]
    target_day_numbers: List[int] = Field(
        default_factory=list,
        description="Day numbers (from the CURRENT daily itinerary) this edit "
        "targets. Required and non-empty when scope='stop'. Empty when "
        "scope='macro' or 'unsupported'.",
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
            "Current daily itinerary (day_number: city): {daily_summary}\n\n"
            "Decide the scope:\n"
            "- 'macro': the request changes the allocation of days across "
            "stops, adds/removes/reorders stops, or otherwise affects more "
            "than one stop (e.g. 'give Tokyo one more day and take it from "
            "Osaka', 'make the whole trip cheaper').\n"
            "- 'stop': the request only changes activities within one stop's "
            "existing days (e.g. 'less museums in Kyoto', 'more food on day "
            "3'). Set target_day_numbers to the exact day_number(s) that stop "
            "occupies.\n"
            "- 'unsupported': the request isn't a plan-edit at all (a factual "
            "question, something unrelated to this trip, or a change this "
            "planner can't make).",
        ),
        ("human", "{edit_request}"),
    ]
)

edit_classify_chain = EDIT_CLASSIFY_PROMPT | edit_classifier


def _validate_stop_target(
    macro_plan: MacroPlan, daily_itinerary: DailyItinerary, target_day_numbers: List[int]
) -> Optional[str]:
    """Returns an error string if target_day_numbers is unusable for a
    whole-stop edit, or None if it resolves cleanly to a single stop."""
    if not target_day_numbers:
        return "no target day(s) were identified for this stop-level edit."

    valid_days = {day.day_number for day in daily_itinerary.days}
    invalid_days = [d for d in target_day_numbers if d not in valid_days]
    if invalid_days:
        return f"day(s) {invalid_days} don't exist in the current itinerary."

    try:
        _find_stop_for_days(macro_plan, target_day_numbers)
    except ValueError:
        return (
            f"day(s) {target_day_numbers} span more than one stop; this kind "
            "of cross-stop change needs a macro-level edit instead."
        )
    return None


def classify_edit(state: EditState) -> dict:
    macro_plan = state["macro_plan"]
    daily_itinerary = state["daily_itinerary"]

    macro_summary = "; ".join(f"{stop.city} ({stop.days}d)" for stop in macro_plan.stops)
    daily_summary = "; ".join(
        f"day {day.day_number}: {day.city}" for day in daily_itinerary.days
    )

    classification = edit_classify_chain.invoke(
        {
            "macro_summary": macro_summary,
            "daily_summary": daily_summary,
            "edit_request": state["edit_request"],
        }
    )

    scope = classification.scope
    target_day_numbers = classification.target_day_numbers or None
    note = classification.note

    if scope == "stop":
        error = _validate_stop_target(macro_plan, daily_itinerary, target_day_numbers or [])
        if error:
            scope = "unsupported"
            note = f"Could not resolve the target day(s) for this edit: {error}"

    return {
        "edit_scope": scope,
        "target_day_numbers": target_day_numbers,
        "classification_note": note,
    }


def _find_stop_for_days(macro_plan: MacroPlan, target_day_numbers: List[int]) -> MacroStop:
    day_number = 1
    for stop in macro_plan.stops:
        stop_days = set(range(day_number, day_number + stop.days))
        if set(target_day_numbers) <= stop_days:
            return stop
        day_number += stop.days
    raise ValueError(
        f"target_day_numbers {target_day_numbers} do not fall entirely within "
        "a single stop; this should have been classified as scope='macro'."
    )


def _splice_stop_days(
    baseline_days: List[DayPlan],
    target_day_numbers: List[int],
    stop: MacroStop,
    new_days: List[StopDayContent],
) -> List[DayPlan]:
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

    instruction = f"Traveler wants this change: {state['edit_request']}."
    if feedback:
        instruction += f" {feedback}"

    candidate_macro_plan = macro_chain.invoke(
        {
            "trip_request": trip_request.model_dump_json(),
            "feedback": instruction,
            "current_plan": current_plan,
        }
    )
    candidate_daily_itinerary = _build_daily_itinerary(trip_request, candidate_macro_plan)
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
    target_day_numbers = state["target_day_numbers"]
    feedback = state.get("validation_feedback")

    # classify_edit has already validated target_day_numbers resolves to a
    # single stop; this call is expected to succeed.
    stop = _find_stop_for_days(macro_plan, target_day_numbers)
    all_stop_day_numbers = _stop_day_numbers(macro_plan, stop)

    previous_activities = [
        activity.name
        for day in baseline_days
        if day.day_number in all_stop_day_numbers
        for activity in day.activities
    ]
    instruction = (
        f"Traveler edit request: '{state['edit_request']}'. "
        f"Previously planned activities for this stop: "
        f"{', '.join(previous_activities) or 'none'}."
    )
    if feedback:
        instruction += f" {feedback}"

    stop_itinerary = _plan_stop(trip_request, stop, extra_instruction=instruction)

    new_days = _splice_stop_days(baseline_days, all_stop_day_numbers, stop, stop_itinerary.days)
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

    total_days = sum(stop.days for stop in macro_plan.stops)
    if total_days != trip_request.duration_days:
        problems.append(f"stop days sum to {total_days}, expected {trip_request.duration_days}")

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

    if not problems:
        return {"validation_feedback": None}

    return {
        "validation_feedback": "Fix these issues: " + "; ".join(problems),
        "edit_retry_count": state["edit_retry_count"] + 1,
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
    macro_plan: MacroPlan, daily_itinerary: DailyItinerary, result: dict
):
    if result["edit_scope"] == "unsupported":
        return macro_plan, daily_itinerary, result["response_message"]
    return result["candidate_macro_plan"], result["candidate_daily_itinerary"], None


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
                "edit_scope": None,
                "target_day_numbers": None,
                "classification_note": None,
                "edit_retry_count": 0,
                "validation_feedback": None,
                "candidate_macro_plan": None,
                "candidate_daily_itinerary": None,
                "response_message": None,
            }
        )
        macro_plan, daily_itinerary, message_out = _apply_committed_edit(
            macro_plan, daily_itinerary, edit_result
        )
        if message_out:
            print(message_out)
        else:
            print("\nUpdated.")
            _print_result({"macro_plan": macro_plan, "daily_itinerary": daily_itinerary})
