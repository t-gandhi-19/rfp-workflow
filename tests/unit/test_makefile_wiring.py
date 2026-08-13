"""The Makefile's env wiring is code, so it gets tests.

WHY THIS EXISTS. Amendment Q's Neo4j auth probe shipped in a state where
`make preflight` could not run it on any machine: the target applied
`PREFLIGHT_ENV` but not `HOST_NEO4J`, so it dialled `bolt://neo4j:7687` — the
compose service name from `.env`, which resolves only inside the compose
network. `make ingest` inherited the failure through
`ingest: preflight-pre-ingest apply-schema` and stopped at the gate, breaking the
clean-clone path CLAUDE.md rule 31 guarantees.

**A fully green suite and a fully green CI were both consistent with that.**
Nothing invoked either target: CI called `python -m scripts.*` with its own
explicit `NEO4J_URI` export, and the probe's unit tests passed the env dict
directly. The recipe — the one line that was wrong — was the only part nothing
executed.

So the drift class is specific: *a target is added or edited without the env
wiring its command needs, and every existing test still passes.* These
assertions read the Makefile as text and catch exactly that. They do not need
make, a stack, or a model, which is what makes them cheap enough to guard every
target rather than the one that broke.

The companion coverage is in CI: the smoke job runs the schema, ingest and
calibrate flows THROUGH their make targets rather than through
`python -m scripts.*`, so those recipes actually execute, and those steps
deliberately no longer export NEO4J_URI themselves — supplying by hand what the
recipe must supply is what hid the fault.

`make preflight` and `make preflight-pre-ingest` stay out of CI scope
permanently: their checks need Ollama, which CI does not run (CLAUDE.md rule
26). The assertions here are the only coverage those two targets can have, which
is exactly why they exist — the fault was in one of them.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE = REPO_ROOT / "Makefile"

#: Make variables carrying host-side address overrides. `.env` holds the
#: CONTAINER view of the world — compose service names — because that is what the
#: containers need; anything invoked from the host has to override it.
HOST_NEO4J = "$(HOST_NEO4J)"
PREFLIGHT_ENV = "$(PREFLIGHT_ENV)"
HOST_PG = "$(HOST_PG)"

#: What each script-invoking target must apply, and why.
#:
#: Every entry is a decision about what its command actually dials, so a target
#: whose command changes has to revisit its row. The `completeness` test below
#: makes this table mandatory rather than best-effort: a new target that invokes
#: a script and is absent from here FAILS, instead of silently going uncovered.
#:
#: Deliberately expressed as REQUIRED fragments, not exact recipes. Asserting the
#: whole recipe would fail on every unrelated edit and teach people to update the
#: expectation without reading it.
REQUIRED_WIRING: dict[str, set[str]] = {
    # Dials Neo4j (amendment Q's probe) AND Ollama/LiteLLM (the model checks).
    "preflight": {HOST_NEO4J, PREFLIGHT_ENV},
    "preflight-pre-ingest": {HOST_NEO4J, PREFLIGHT_ENV},
    # Dials Neo4j only.
    "apply-schema": {HOST_NEO4J},
    "ingest": {HOST_NEO4J},
    "reembed": {HOST_NEO4J},
    # Dials the gateway and the write-api, never Neo4j: it reads fixtures from
    # disk. HOST_NEO4J here would imply a dependency that does not exist.
    "calibrate": {PREFLIGHT_ENV},
    "calibrate-dry": {PREFLIGHT_ENV},
    "calibrate-commission": {PREFLIGHT_ENV},
    # Postgres over the published port, via Alembic.
    "migrate": {HOST_PG},
    "migrate-status": {HOST_PG},
    # Neither: they read and write only the repo.
    "fixtures": set(),
    "fixtures-check": set(),
    "validate-manual-key": set(),
    "manual-key-schema": set(),
}

#: Targets that dial Neo4j from the host. Named separately from the table above
#: so the rule is legible on its own: this is the exact set the shipped fault
#: belonged to.
DIALS_NEO4J_FROM_HOST = {"preflight", "preflight-pre-ingest", "apply-schema", "ingest", "reembed"}


def read_makefile() -> str:
    return MAKEFILE.read_text(encoding="utf-8")


def recipes() -> dict[str, str]:
    """Every target's recipe body, keyed by target name.

    A recipe is the tab-indented block following a `target:` line. Continuations
    (`\\` at end of line) are already part of the block, so joining the lines is
    enough — the assertions only ever look for substrings.
    """
    blocks: dict[str, str] = {}
    current: str | None = None
    lines: list[str] = []
    for line in read_makefile().splitlines():
        if line.startswith("\t"):
            if current is not None:
                lines.append(line)
            continue
        if current is not None:
            blocks[current] = "\n".join(lines)
            current, lines = None, []
        match = re.match(r"^([A-Za-z0-9_.-]+)\s*:(?!=)", line)
        if match:
            current, lines = match.group(1), []
    if current is not None:
        blocks[current] = "\n".join(lines)
    return blocks


def script_invoking_targets() -> dict[str, str]:
    """Targets whose recipe runs a `python -m scripts.*` module."""
    return {
        name: body for name, body in recipes().items() if re.search(r"python -m scripts\.", body)
    }


class TestRecipeParsing:
    """The parser has to be right, or every assertion below is vacuous."""

    def test_the_makefile_is_parsed_into_targets(self) -> None:
        parsed = recipes()
        assert "preflight" in parsed
        assert "ingest" in parsed
        # A variable assignment is not a target, despite containing a colon.
        assert "HOST_NEO4J" not in parsed
        assert "PREFLIGHT_ENV" not in parsed

    def test_recipes_carry_their_commands(self) -> None:
        assert "scripts.preflight" in recipes()["preflight"]
        assert "scripts.ingest" in recipes()["ingest"]

    def test_script_invoking_targets_are_found(self) -> None:
        found = set(script_invoking_targets())
        assert {"preflight", "ingest", "apply-schema", "calibrate"} <= found
        # `up` runs compose, not a script.
        assert "up" not in found


class TestHostAddressWiring:
    """The assertion the shipped fault would have failed."""

    @pytest.mark.parametrize("target", sorted(DIALS_NEO4J_FROM_HOST))
    def test_every_target_that_dials_neo4j_from_the_host_overrides_the_uri(
        self, target: str
    ) -> None:
        """`.env` sets NEO4J_URI to the compose service name.

        It resolves inside the compose network and nowhere else, so a host-side
        target without HOST_NEO4J dials a name that cannot exist. That is
        precisely what `make preflight` and `make preflight-pre-ingest` did.
        """
        body = recipes()[target]
        assert HOST_NEO4J in body, (
            f"'{target}' runs from the host and dials Neo4j, but its recipe does not apply "
            f"{HOST_NEO4J}. Without it the target uses .env's NEO4J_URI, which names a "
            f"compose service and does not resolve from the host."
        )

    @pytest.mark.parametrize(
        "target", sorted(t for t, w in REQUIRED_WIRING.items() if PREFLIGHT_ENV in w)
    )
    def test_targets_reaching_ollama_or_litellm_use_the_host_urls(self, target: str) -> None:
        """Containers reach Ollama at `host.docker.internal`, which does not
        resolve from the host shell — so the host forms are separate variables."""
        assert PREFLIGHT_ENV in recipes()[target]

    def test_calibrate_does_not_claim_a_neo4j_dependency_it_does_not_have(self) -> None:
        """Wiring is a claim about what a command touches.

        `scripts.calibrate` reads fixtures from disk and calls the gateway and
        the write-api. It never opens a graph session, so HOST_NEO4J on it would
        assert a dependency that does not exist and would make the table above
        stop meaning anything.
        """
        for target in ("calibrate", "calibrate-dry", "calibrate-commission"):
            assert HOST_NEO4J not in recipes()[target]

    def test_every_required_fragment_is_present(self) -> None:
        """The table, applied in full."""
        parsed = recipes()
        missing = {
            target: sorted(fragment for fragment in required if fragment not in parsed[target])
            for target, required in REQUIRED_WIRING.items()
            if target in parsed and any(fragment not in parsed[target] for fragment in required)
        }
        assert not missing, f"targets missing required env wiring: {missing}"


class TestCoverageIsComplete:
    """The half that catches a target nobody thought about."""

    def test_every_script_invoking_target_has_a_recorded_wiring_decision(self) -> None:
        """A new target must state what it dials, or fail here.

        This is the drift class that shipped: the fault was not a wrong entry in
        a table, it was a target whose env needs nobody had written down. An
        assertion over a hand-listed set of targets would have had the same blind
        spot, so the set is derived from the Makefile instead.
        """
        undeclared = sorted(set(script_invoking_targets()) - set(REQUIRED_WIRING))
        assert not undeclared, (
            f"these targets run a script but have no wiring decision recorded: {undeclared}. "
            f"Add each to REQUIRED_WIRING with the fragments its command needs — an empty set "
            f"if it dials nothing — so the choice is explicit rather than assumed."
        )

    def test_the_table_does_not_name_targets_that_no_longer_exist(self) -> None:
        """A stale row asserts nothing while looking like it asserts something."""
        parsed = recipes()
        stale = sorted(target for target in REQUIRED_WIRING if target not in parsed)
        assert not stale, f"REQUIRED_WIRING names targets absent from the Makefile: {stale}"


class TestExpansion:
    """`make -n` proves the variables expand to real assignments.

    The text assertions above prove the fragment is PRESENT. They cannot prove it
    expands to anything useful — a HOST_NEO4J defined as the empty string would
    satisfy every one of them. This closes that gap, and needs make.

    Skipped where make is absent (the Windows build host has none) and run in
    CI's ubuntu job, so the stronger check exists without becoming the only one.
    """

    @pytest.fixture(autouse=True)
    def make_path(self) -> str:
        resolved = shutil.which("make")
        if resolved is None:
            pytest.skip("make is not installed; the text assertions above still apply")
        return resolved

    def dry_run(self, target: str) -> str:
        resolved = shutil.which("make")
        assert resolved is not None  # the fixture skipped otherwise
        # Absolute path, fixed argv, no shell, and `target` comes from this
        # module's own constants — never from input.
        result = subprocess.run(  # noqa: S603
            [resolved, "-n", target],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout + result.stderr

    @pytest.mark.parametrize("target", sorted(DIALS_NEO4J_FROM_HOST))
    def test_the_expanded_recipe_sets_neo4j_uri_to_a_host_address(self, target: str) -> None:
        printed = self.dry_run(target)
        assert "NEO4J_URI=bolt://localhost:" in printed, (
            f"'make -n {target}' does not expand to a localhost NEO4J_URI assignment. "
            f"HOST_NEO4J is either missing from the recipe or no longer defines one."
        )

    @pytest.mark.parametrize(
        "target", sorted(t for t, w in REQUIRED_WIRING.items() if PREFLIGHT_ENV in w)
    )
    def test_the_expanded_recipe_sets_the_host_model_urls(self, target: str) -> None:
        printed = self.dry_run(target)
        assert "OLLAMA_BASE_URL_HOST=" in printed
        assert "LITELLM_BASE_URL_HOST=" in printed
