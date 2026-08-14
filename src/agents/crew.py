"""crewAI, wired the way this repository's rules require.

WHAT IS SWITCHED OFF, AND WHY EACH ONE MATTERS.

* **Memory (rule 6).** `Crew(memory=False)` and no knowledge sources or
  embedder on any agent. State is typed Pydantic checkpointed to Postgres and
  the knowledge graph is the only cross-run memory. A framework that quietly
  remembered the last run would make a run's output depend on history nobody
  can see in the run record.
* **Manager and delegation (rule 7).** No `manager_agent`, no `manager_llm`, no
  `planning`, and `allow_delegation=False` on every agent. Stage order is the
  controller's, in Python.
* **Loops (rule 14).** `max_iter=1`, `max_retry_limit=0`, `Task(max_retries=0)`.
  The hard call budget permits one draft and one critique; a framework free to
  iterate would spend against a ceiling the controller is required not to
  delegate. The ONE permitted retry is the controller's, carries the validation
  error, and is counted in its ledger.

WHY EVERY AGENT IS BUILT AGAINST THE PROXY. Rule 5. crewAI's model client is
`litellm` the LIBRARY, which would talk to Groq and Ollama directly given
provider credentials. Every `LLM` here is constructed with the LiteLLM PROXY's
base_url and an `openai/`-prefixed alias, so the proxy stays the single place
the alias table, the retry policy, the ceilings and cost accounting live. The
AST guard in `tests/security/` cannot see this — it reads our imports, not a
dependency's behaviour — so `tests/unit/test_agent_wiring.py` asserts it.
"""

from __future__ import annotations

import os
from typing import Any

from crewai import LLM, Agent, Crew, Process, Task
from pydantic import BaseModel

from src.controller.limits import limits_config
from src.gateway.client import gateway_base_url
from src.prompts import Prompt

#: litellm routes an `openai/`-prefixed model to whatever `base_url` says, which
#: is how an alias defined in `docker/litellm/config.yaml` is reached without
#: naming a provider anywhere above the gateway.
PROXY_PREFIX = "openai/"


class AgentWiringError(RuntimeError):
    """The crew was constructed in a way one of the rules forbids."""


def build_llm(prompt: Prompt) -> LLM:
    """One LLM, pinned to the proxy and to the alias the prompt declares.

    The alias comes from the PROMPT FILE rather than from a caller, because a
    drafter prompt sent to the triage model is a silent quality regression and
    the file is the only place that pairing is recorded (rule 17).
    """
    timeouts = limits_config().gateway.timeouts_seconds
    if prompt.model_alias not in timeouts:
        raise AgentWiringError(
            f"prompt '{prompt.name}' names alias '{prompt.model_alias}', which has no "
            f"timeout in config/limits.yaml. Add one rather than defaulting: an alias "
            f"with no declared timeout is one nobody has decided how long to wait for."
        )
    return LLM(
        model=f"{PROXY_PREFIX}{prompt.model_alias}",
        base_url=f"{gateway_base_url().rstrip('/')}/v1",
        api_key=os.environ.get("LITELLM_MASTER_KEY", ""),
        temperature=prompt.temperature if prompt.temperature is not None else 0.0,
        timeout=timeouts[prompt.model_alias],
    )


def build_agent(
    *,
    role: str,
    goal: str,
    backstory: str,
    prompt: Prompt,
    tools: list[Any] | None = None,
) -> Agent:
    """One agent, with every framework autonomy this project forbids disabled."""
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=build_llm(prompt),
        tools=tools or [],
        # Rule 7: no delegation. Stage order is the controller's.
        allow_delegation=False,
        # Rule 14: one pass. The controller owns the single permitted retry.
        max_iter=1,
        max_retry_limit=0,
        # Rule 6: no memory of any kind, and nothing that would create one.
        knowledge_sources=None,
        embedder=None,
        cache=False,
        reasoning=False,
        verbose=False,
    )


def build_task(
    *,
    description: str,
    expected_output: str,
    agent: Agent,
    output_model: type[BaseModel],
) -> Task:
    """One task. `output_pydantic` is mandatory (rule 13).

    `output_model` has no default and is not optional, so there is no way to
    construct a task here that returns a bare dict across a module boundary.
    """
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
        output_pydantic=output_model,
        # The controller's retry is the only retry; see the module docstring.
        #
        # THIS DEFAULTS TO 3. Left alone, a task whose output failed validation
        # would be re-issued three times by the framework — three model calls
        # the ledger never sees, against a ceiling rule 14 says the controller
        # must not delegate. The deprecated spelling of this field is
        # `max_retries`; setting that one instead earns a DeprecationWarning and
        # will silently stop working at crewAI v1.
        guardrail_max_retries=0,
        async_execution=False,
        human_input=False,
    )


def build_crew(*, agent: Agent, task: Task) -> Crew:
    """A one-agent, one-task crew.

    Deliberately not a multi-agent crew. Each of the five steps is invoked by
    the controller at the point its stage says to, so the framework never gets
    to decide what runs next — which is rule 7 stated as a construction rather
    than as a comment.
    """
    return Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        # Rule 6, all four switches. crewAI spells memory in several places and
        # leaving any one of them at its default would leave a memory on.
        memory=False,
        long_term_memory=None,
        short_term_memory=None,
        entity_memory=None,
        external_memory=None,
        # Rule 7. Naming these explicitly is the point: a manager appears the
        # moment `Process.hierarchical` is set, and it needs one of these.
        manager_agent=None,
        manager_llm=None,
        planning=False,
        cache=False,
        verbose=False,
    )
