"""rfp-workflow — agentic RFP response workflow (cloud migration domain).

`src` is the single package root, so every module is imported as
``src.<package>...``. That keeps the paths in CLAUDE.md and the build prompt
literally true and makes the CI guard "no provider SDK imports outside
src/gateway/" an exact path check rather than a heuristic.
"""
