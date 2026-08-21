"""Phase 1: natural-language trip intake with clarification for missing essentials."""

from typing import Optional

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

ESSENTIAL_FIELDS = {
    "duration_days": "trip duration in days",
    "budget_usd": "total budget in US dollars",
    "traveler_count": "number of travelers",
}


class TripRequest(BaseModel):
    """Structured representation of a trip the user is planning."""

    destination: Optional[str] = Field(
        default=None,
        description="Destination, region, or country. May be vague or absent.",
    )
    duration_days: Optional[int] = Field(
        default=None, description="Trip length in days."
    )
    budget_usd: Optional[float] = Field(
        default=None, description="Total trip budget in US dollars."
    )
    traveler_count: Optional[int] = Field(
        default=None,
        description="Number of travelers, e.g. 'me and my girlfriend' -> 2.",
    )
    interests: Optional[list[str]] = Field(
        default=None, description="Stated interests, e.g. ['food', 'anime']."
    )
    pace: Optional[str] = Field(
        default=None, description="Desired pace, e.g. relaxed, moderate, packed."
    )
    notes: Optional[str] = Field(
        default=None, description="Any other relevant detail that doesn't fit above."
    )


model = ChatOpenAI(model="gpt-4o-mini")
extractor = model.with_structured_output(TripRequest)

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract trip-planning details into a structured TripRequest.\n"
            "You are given the currently known trip details (as JSON) and a new "
            "message from the user. Return the FULL updated TripRequest:\n"
            "- Keep every field the user did not address unchanged.\n"
            "- Update or fill in fields the user did address.\n"
            "- Convert phrases like 'me and my girlfriend' or 'my wife and I' into "
            "traveler_count (e.g. 2).\n"
            "- Never invent information the user didn't state or imply.",
        ),
        ("system", "Known so far: {known_state}"),
        ("human", "{message}"),
    ]
)

extraction_chain = EXTRACTION_PROMPT | extractor

FOLLOWUP_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a friendly travel planning assistant. The user is describing "
            "a trip, and the following essential details are still missing: "
            "{missing_fields}. Ask a single, natural, concise question requesting "
            "just this missing information. Do not repeat information already known.",
        ),
    ]
)

followup_chain = FOLLOWUP_PROMPT | model


def missing_essentials(trip: TripRequest) -> list[str]:
    return [
        description
        for field, description in ESSENTIAL_FIELDS.items()
        if getattr(trip, field) is None
    ]


def run_intake() -> TripRequest:
    trip = TripRequest()
    print("Tell me about the trip you want to plan.")

    while True:
        message = input("> ").strip()
        if not message:
            continue

        trip = extraction_chain.invoke(
            {"known_state": trip.model_dump_json(), "message": message}
        )

        missing = missing_essentials(trip)
        if not missing:
            break

        question = followup_chain.invoke({"missing_fields": ", ".join(missing)})
        print(question.content)

    print("\nGot everything I need:")
    print(trip.model_dump_json(indent=2))
    return trip


if __name__ == "__main__":
    run_intake()
