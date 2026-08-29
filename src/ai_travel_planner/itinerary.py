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
            "- Give a short rationale for each stop.",
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


def _plan_stop(trip_request: TripRequest, stop: MacroStop) -> StopItinerary:
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


def generate_daily_itinerary(state: ItineraryState) -> dict:
    trip_request = state["trip_request"]
    macro_plan = state["macro_plan"]

    days: List[DayPlan] = []
    day_number = 1
    total_cost = 0.0
    have_cost = False

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
            for activity in day_content.activities:
                if activity.estimated_cost_usd is not None:
                    total_cost += activity.estimated_cost_usd
                    have_cost = True
            day_number += 1

    daily_itinerary = DailyItinerary(
        days=days,
        estimated_total_cost_usd=total_cost if have_cost else None,
    )
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
    _print_result(result)
