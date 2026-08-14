"""The agent crew (1c).

Backed by crewAI, with memory off, no manager, no delegation and no loops —
every one of those disabled by construction in `src.agents.crew`, not by
convention. The five steps are invoked by the deterministic controller, which
owns stage order and the call budget.
"""

from __future__ import annotations

from src.agents.crew import AgentWiringError, build_agent, build_crew, build_llm, build_task
from src.agents.layer import CrewAgentLayer, build_selection, to_drafted_answer
from src.agents.mcp_client import McpClient, McpError, drafter_client, retriever_client

__all__ = [
    "AgentWiringError",
    "CrewAgentLayer",
    "McpClient",
    "McpError",
    "build_agent",
    "build_crew",
    "build_llm",
    "build_selection",
    "build_task",
    "drafter_client",
    "retriever_client",
    "to_drafted_answer",
]
