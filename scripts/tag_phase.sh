#!/usr/bin/env bash
# Create and push a phase tag, refusing unless it is safe to do so.
#
# Context: v0.2 was once tagged against a `main` that had not actually received
# the phase branch, so the tag pointed at the previous phase's code. It was
# caught and corrected, but "caught" is luck. Three preconditions make that
# specific mistake impossible:
#
#   1. HEAD is on main            — a tag on a feature branch names the wrong tree
#   2. the worktree is clean      — otherwise the tag names something no one has
#   3. main == origin/main        — the local branch must already have the merge,
#                                   which is exactly the check that would have
#                                   failed last time
#
# Phase tags are only ever created through this script.
set -euo pipefail

TAG="${TAG:-}"

fail() {
    printf 'refusing to tag: %s\n' "$1" >&2
    exit 1
}

[ -n "$TAG" ] || fail "TAG is not set. Usage: make tag-phase TAG=v0.3"

case "$TAG" in
    v[0-9]*.[0-9]*) ;;
    *) fail "'$TAG' does not look like a phase tag (expected vMAJOR.MINOR, e.g. v0.3)" ;;
esac

branch="$(git rev-parse --abbrev-ref HEAD)"
[ "$branch" = "main" ] || fail "HEAD is on '$branch', not main. Phase tags name a merged main."

if [ -n "$(git status --porcelain)" ]; then
    git status --short >&2
    fail "the worktree is dirty. A tag must name a tree someone can actually check out."
fi

git fetch --quiet origin main

local_sha="$(git rev-parse main)"
remote_sha="$(git rev-parse origin/main)"
if [ "$local_sha" != "$remote_sha" ]; then
    fail "main ($(git rev-parse --short main)) and origin/main ($(git rev-parse --short origin/main)) differ.
    Pull or push first — this is the check that catches tagging a main that has
    not yet received the phase merge."
fi

if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
    fail "$TAG already exists locally. Delete it deliberately if it is wrong."
fi
if git ls-remote --exit-code --tags origin "refs/tags/$TAG" >/dev/null 2>&1; then
    fail "$TAG already exists on origin. Delete it deliberately if it is wrong."
fi

subject="$(git log -1 --pretty=%s)"
printf 'tagging %s at %s\n  %s\n' "$TAG" "$(git rev-parse --short main)" "$subject"

git tag -a "$TAG" -m "$TAG"
git push --quiet origin "$TAG"
printf 'pushed %s\n' "$TAG"
