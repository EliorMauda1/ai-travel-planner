"""Phase 2: LangGraph workflow that turns a TripRequest into a macro + daily itinerary."""

from typing import List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from ai_travel_planner.intake import TripRequest, run_intake

load_dotenv()

MAX_MACRO_RETRIES = 2

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


class DayPlan(BaseModel):
    day_number: int
    city: str
    activities: List[Activity]


class DailyItinerary(BaseModel):
    days: List[DayPlan]
    estimated_total_cost_usd: Optional[float] = None


class StopDayContent(BaseModel):
    activities: List[Activity]


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


# --- Daily itinerary generation ------------------------------------------

day_extractor = model.with_structured_output(StopItinerary)

DAILY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a travel planner creating a day-by-day itinerary for one "
            "stop of a larger trip.\n"
            "Create exactly {num_days} day(s) of activities for {city}.\n"
            "Take the traveler's interests and pace preference into account. "
            "Each day should have 2-4 activities (sightseeing, food, nature, "
            "culture, etc.) appropriate to the stated pace, with a rough "
            "estimated_cost_usd per activity where reasonable.",
        ),
        ("system", "Trip context: {trip_request}"),
        ("system", "This stop: {city}, {num_days} day(s). Rationale: {rationale}"),
    ]
)

day_chain = DAILY_PROMPT | day_extractor


def generate_daily_itinerary(state: ItineraryState) -> dict:
    trip_request = state["trip_request"]
    macro_plan = state["macro_plan"]

    days: List[DayPlan] = []
    day_number = 1
    total_cost = 0.0
    have_cost = False

    for stop in macro_plan.stops:
        stop_itinerary = day_chain.invoke(
            {
                "trip_request": trip_request.model_dump_json(),
                "city": stop.city,
                "num_days": stop.days,
                "rationale": stop.rationale,
            }
        )
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
            print(f"  - [{activity.category}] {activity.name}{cost}: {activity.description}")

    if daily_itinerary.estimated_total_cost_usd is not None:
        print(f"\nEstimated total cost: ${daily_itinerary.estimated_total_cost_usd:.0f}")


if __name__ == "__main__":
    trip = run_intake()
    print("\nGenerating itinerary...")
    result = plan_trip(trip)
    _print_result(result)
