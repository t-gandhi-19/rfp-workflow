"""crewAI is configured the way the rules require — asserted, not assumed.

WHY THIS FILE HAS TO EXIST. The AST guard in `tests/security/` proves that no
module outside `src/gateway/` IMPORTS a provider SDK. It cannot prove anything
about what a DEPENDENCY does at runtime, and crewAI's model client is `litellm`
the library, which will talk to Groq and Ollama directly given credentials. Rule
5 therefore has a hole exactly the width of this dependency, and these
assertions are what fill it.

The same is true of rules 6, 7 and 14: memory, managers and loops are all
crewAI defaults or one keyword away, and every one of them is disabled in
`src/agents/crew.py` by construction. A future edit that re-enables any of them
fails here rather than in a run nobody audits.
"""

from __future__ import annotations

import pytest

from src.agents.crew import PROXY_PREFIX, AgentWiringError, build_agent, build_crew, build_task
from src.contracts import CritiqueResult, DraftPayload, TriageResult
from src.prompts import Prompt, load_prompt

#: Every prompt that drives an agent, and the contract its task returns.
AGENT_PROMPTS = [
    ("triage", TriageResult),
    ("drafter", DraftPayload),
    ("critic", CritiqueResult),
]


def agent_for(name: str):  # type: ignore[no-untyped-def]
    prompt = load_prompt(name)
    return build_agent(role="r", goal="g", backstory="b", prompt=prompt), prompt


def crew_for(name: str, model: type):  # type: ignore[no-untyped-def]
    agent, _ = agent_for(name)
    task = build_task(description="d", expected_output="e", agent=agent, output_model=model)
    return build_crew(agent=agent, task=task), agent, task


class TestEveryCallLeavesThroughTheProxy:
    """Rule 5, for the one dependency the AST guard cannot see."""

    @pytest.mark.parametrize(("name", "model"), AGENT_PROMPTS)
    def test_the_model_is_an_alias_not_a_provider(self, name: str, model: type) -> None:
        """`openai/drafter-model`, never `groq/llama-3.3-70b-versatile`.

        The prefix is how litellm is told "this is an OpenAI-compatible endpoint
        at base_url"; the suffix is an alias defined in the gateway config. A
        provider name here would route around the proxy entirely.
        """
        agent, prompt = agent_for(name)
        assert agent.llm.model == f"{PROXY_PREFIX}{prompt.model_alias}"

    @pytest.mark.parametrize(("name", "model"), AGENT_PROMPTS)
    def test_the_base_url_is_the_gateway(self, name: str, model: type) -> None:
        from src.gateway.client import gateway_base_url

        agent, _ = agent_for(name)
        assert agent.llm.base_url.startswith(gateway_base_url().rstrip("/"))

    @pytest.mark.parametrize(("name", "model"), AGENT_PROMPTS)
    def test_no_provider_name_appears_anywhere_in_the_model_string(
        self, name: str, model: type
    ) -> None:
        agent, _ = agent_for(name)
        lowered = agent.llm.model.lower()
        for provider in ("groq", "ollama", "anthropic", "cohere", "mistral", "vertex"):
            assert provider not in lowered, f"{name} names the provider '{provider}'"


class TestTheAliasComesFromThePromptFile:
    """Rule 17. A drafter prompt sent to the triage model is a silent quality
    regression, and the file is the only place the pairing is recorded."""

    def test_the_drafter_agent_uses_the_drafter_alias(self) -> None:
        agent, _ = agent_for("drafter")
        assert agent.llm.model.endswith("drafter-model")

    def test_the_triage_agent_uses_the_triage_alias(self) -> None:
        agent, _ = agent_for("triage")
        assert agent.llm.model.endswith("triage-model")

    def test_an_alias_with_no_declared_timeout_is_refused(self) -> None:
        """Not defaulted. An alias nobody has decided how long to wait for is a
        decision that has not been made, and a silent default makes it look
        like it has."""
        rogue = Prompt(
            name="rogue",
            version=1,
            model_alias="a-model-nobody-configured",
            body="body",
            path=load_prompt("triage").path,
        )
        with pytest.raises(AgentWiringError, match="no timeout"):
            build_agent(role="r", goal="g", backstory="b", prompt=rogue)

    def test_the_timeout_comes_from_limits_yaml(self) -> None:
        from src.controller.limits import limits_config

        agent, prompt = agent_for("drafter")
        assert agent.llm.timeout == limits_config().gateway.timeouts_seconds[prompt.model_alias]


class TestMemoryIsOff:
    """Rule 6. crewAI spells memory in five places; leaving any one at its
    default would leave a memory on."""

    @pytest.mark.parametrize(("name", "model"), AGENT_PROMPTS)
    def test_the_crew_has_no_memory(self, name: str, model: type) -> None:
        crew, _, _ = crew_for(name, model)
        assert crew.memory is False

    @pytest.mark.parametrize(
        "attribute",
        ["long_term_memory", "short_term_memory", "entity_memory", "external_memory"],
    )
    def test_no_memory_store_of_any_kind_is_attached(self, attribute: str) -> None:
        crew, _, _ = crew_for("triage", TriageResult)
        assert getattr(crew, attribute) is None

    @pytest.mark.parametrize(("name", "model"), AGENT_PROMPTS)
    def test_no_agent_carries_knowledge_or_an_embedder(self, name: str, model: type) -> None:
        """Both are memory by another name: either one would let a run's output
        depend on history that does not appear in the run record."""
        agent, _ = agent_for(name)
        assert agent.knowledge_sources is None
        assert agent.embedder is None


class TestThereIsNoManager:
    """Rule 7. The execution controller is deterministic Python."""

    @pytest.mark.parametrize(("name", "model"), AGENT_PROMPTS)
    def test_no_manager_agent_and_no_manager_llm(self, name: str, model: type) -> None:
        crew, _, _ = crew_for(name, model)
        assert crew.manager_agent is None
        assert crew.manager_llm is None

    @pytest.mark.parametrize(("name", "model"), AGENT_PROMPTS)
    def test_the_process_is_sequential(self, name: str, model: type) -> None:
        """`Process.hierarchical` is what introduces a manager, and it is one
        keyword away."""
        crew, _, _ = crew_for(name, model)
        assert crew.process.value == "sequential"

    @pytest.mark.parametrize(("name", "model"), AGENT_PROMPTS)
    def test_planning_is_off(self, name: str, model: type) -> None:
        """A planning LLM deciding what runs next is a supervisor with a
        different name."""
        crew, _, _ = crew_for(name, model)
        assert crew.planning is False

    @pytest.mark.parametrize(("name", "model"), AGENT_PROMPTS)
    def test_no_agent_may_delegate(self, name: str, model: type) -> None:
        agent, _ = agent_for(name)
        assert agent.allow_delegation is False

    @pytest.mark.parametrize(("name", "model"), AGENT_PROMPTS)
    def test_each_crew_holds_exactly_one_agent_and_one_task(self, name: str, model: type) -> None:
        """The controller invokes each step at the point its stage says to, so
        the framework never gets to choose what runs next."""
        crew, _, _ = crew_for(name, model)
        assert len(crew.agents) == 1
        assert len(crew.tasks) == 1


class TestThereAreNoLoops:
    """Rule 14. The hard call budget permits one draft and one critique."""

    @pytest.mark.parametrize(("name", "model"), AGENT_PROMPTS)
    def test_an_agent_takes_one_pass(self, name: str, model: type) -> None:
        agent, _ = agent_for(name)
        assert agent.max_iter == 1

    @pytest.mark.parametrize(("name", "model"), AGENT_PROMPTS)
    def test_the_framework_retries_nothing(self, name: str, model: type) -> None:
        """The ONE permitted retry is the controller's: it carries the
        validation error and is counted in the ledger. A framework retry would
        be an uncounted call against a ceiling the controller must not delegate.
        """
        agent, _ = agent_for(name)
        assert agent.max_retry_limit == 0

    @pytest.mark.parametrize(("name", "model"), AGENT_PROMPTS)
    def test_the_task_retries_nothing_either(self, name: str, model: type) -> None:
        _, _, task = crew_for(name, model)
        assert task.guardrail_max_retries == 0

    def test_the_framework_default_would_have_been_three(self) -> None:
        """The reason the line above is not redundant.

        crewAI re-issues a task whose output failed validation three times by
        default — three model calls the ledger never sees. Asserting the
        DEFAULT here means that if a future version changes it, this test says
        so rather than the setting quietly becoming a no-op that nobody
        re-reads.
        """
        from crewai import Task

        assert Task(description="d", expected_output="e").guardrail_max_retries == 3

    @pytest.mark.parametrize(("name", "model"), AGENT_PROMPTS)
    def test_nothing_is_cached(self, name: str, model: type) -> None:
        """A cache hit is a call the ledger never sees, which makes the recorded
        cost of a run smaller than the run."""
        crew, agent, _ = crew_for(name, model)
        assert crew.cache is False
        assert agent.cache is False


class TestEveryTaskIsTyped:
    """Rule 13: `output_pydantic` on every task, no bare dict across a boundary."""

    @pytest.mark.parametrize(("name", "model"), AGENT_PROMPTS)
    def test_the_task_declares_its_contract(self, name: str, model: type) -> None:
        _, _, task = crew_for(name, model)
        assert task.output_pydantic is model

    def test_the_output_model_cannot_be_omitted(self) -> None:
        """No default, so there is no way to build a task here that returns a
        bare dict."""
        agent, _ = agent_for("triage")
        with pytest.raises(TypeError):
            build_task(description="d", expected_output="e", agent=agent)  # type: ignore[call-arg]

    def test_the_drafter_contract_cannot_carry_a_confidence(self) -> None:
        """§10: confidence is computed arithmetic. A model asked for a number
        supplies a fluent one, and it would sit next to the real one with
        nothing marking which is which."""
        assert "confidence" not in DraftPayload.model_fields

    def test_the_critic_contract_bounds_the_delta_at_zero(self) -> None:
        field = CritiqueResult.model_fields["confidence_delta"]
        assert 0.0 in [getattr(m, "le", None) for m in field.metadata]
