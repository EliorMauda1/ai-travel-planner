# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

AI Travel Planner — a conversational trip-planning assistant, built as both a portfolio project and a learning project for LangChain/LangGraph and agentic AI engineering. Full product plan (MVP scope, phased roadmap, risks) lives outside the repo at `C:\Users\elior\.claude\plans\ai-travel-planner-dapper-metcalfe.md`.

Language: Python (>=3.11), managed via `pyproject.toml` + a virtualenv.

## Working process (read before making changes)

This is a learning project first. For every new phase of work, Claude Code must:

1. Explain the goal of the phase.
2. Explain the relevant LangChain/LangGraph concepts before writing code.
3. Propose a concrete implementation approach.
4. Stop and get explicit approval before writing any code — never decide architecture, frameworks, or libraries silently; always present options + trade-offs + a recommendation.
5. Implement only the approved phase (never multiple phases at once).
6. Verify (run it / test it).
7. Explain what was built and why it works.
8. Stop and wait for approval before starting the next phase.

Git is part of every phase. At the end of each completed phase: verify it works, show the changes, confirm no secrets are staged (especially `.env` / API keys), create one meaningful commit, wait for approval, then push to `origin` and verify the push. Don't create incidental commits for small changes — one clean commit per completed phase.

## Setup

```
pip install -e .
```

Copy `.env.example` to `.env` and fill in:
- `OPENAI_API_KEY`
- `FOURSQUARE_API_KEY` (Foursquare Places API — used by Phase 3's places-grounding tool)

## Architecture

- `src/ai_travel_planner/main.py` — Phase 0 hello-world LangChain/OpenAI script. Kept as-is, not wired into the rest of the app.
- `src/ai_travel_planner/intake.py` — Phase 1. Natural-language trip intake.
  - `TripRequest` (Pydantic): structured trip fields (`destination`, `duration_days`, `budget_usd`, `traveler_count`, `interests`, `pace`, `notes`); all optional except that `duration_days`, `budget_usd`, `traveler_count` are treated as essential.
  - `extraction_chain`: merges each new user message into the current `TripRequest` (via `with_structured_output`), preserving previously-known fields.
  - `followup_chain`: generates a natural follow-up question for whatever essential fields are still missing.
  - `run_intake()`: CLI loop that repeats extraction + follow-up until all essentials are known; returns the final `TripRequest`.
- `src/ai_travel_planner/itinerary.py` — Phases 2 & 3. LangGraph workflow that turns a `TripRequest` into a full itinerary.
  - **Macro stage** (Phase 2): `generate_macro_plan` → `validate_macro_plan` → conditional edge — retries macro generation (capped at `MAX_MACRO_RETRIES`) if the stops' `days` don't sum to `duration_days`, otherwise proceeds. Vague/absent destinations are resolved to concrete cities by the LLM itself; the graph never blocks to ask for clarification.
  - **Per-stop agent sub-flow** (Phase 3): for each macro stop, a small `agent → tools → finalize` LangGraph (`StopPlanningState`) lets the model call the `search_places` tool (backed by the Foursquare Places API) to ground activities/restaurants in real places, capped at `MAX_TOOL_ROUND_TRIPS` round trips. Every `Activity` carries `source: Literal["foursquare", "llm_estimate"]`; the finalize step is instructed to only mark a place `"foursquare"` if it actually came from a tool result — LLM guesses (search failed, no results, or unused) must be labeled `"llm_estimate"`, so unverified content is never presented as API-confirmed. `_print_result` reflects this with `[Foursquare-verified]` vs `[LLM estimate - unverified]` tags.
  - `plan_trip(trip_request)`: runs the full top-level graph (`generate_macro_plan → validate_macro_plan → generate_daily_itinerary → END`).
  - Note: Foursquare's current API lives at `https://places-api.foursquare.com/places/search` and requires `Authorization: Bearer <key>` plus an `X-Places-Api-Version` header — the legacy `api.foursquare.com/v3` key-only endpoint is retired (410s).

Run the full flow end-to-end:

```
python -m ai_travel_planner.itinerary
```

(This runs intake, then macro + daily itinerary generation, then prints the result.)

## Conventions

- Structured LLM output goes through Pydantic models + `with_structured_output()`, not manual JSON parsing.
- LangGraph is used only where there's genuine multi-step/branching/tool-calling behavior worth modeling as a graph (macro validation retry loop; per-stop tool-calling agent). Plain LCEL chains (`prompt | model`) are used everywhere else (intake extraction/follow-up, macro generation, final structured extraction) — don't reach for LangGraph by default.
- Secrets live only in `.env` (git-ignored); never hardcode or commit API keys. `.env.example` documents required variable names with empty values.

## Status / roadmap

Completed: Phase 0 (scaffolding + hello-world), Phase 1 (intake), Phase 2 (macro + daily itinerary via LangGraph), Phase 3 (Foursquare tool-calling grounding).

Next up: Phase 4 — conversational editing of existing plans (the MVP's centerpiece). Do not start it without explicit user approval, and follow the working process above.
