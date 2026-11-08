"""Leave a SWE-bench-format repository only its base commit's history.

Some published images keep the repository's later commits in `/testbed`, through
remotes, branches or tags. `base_history_only` runs before the agent and keeps one
branch at the base commit and the tags that commit contains, as SWE-bench's own
image build does. The working tree and index are unchanged.
"""

from __future__ import annotations

import re

from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import sandbox

# `bash -c HISTORY_SCRUB scrub <base> [repo]`
HISTORY_SCRUB = r"""
set -euo pipefail
cd "${2:-/testbed}"
git config --global --add safe.directory "$PWD"
base=$(git rev-parse --verify --quiet "$1^{commit}")
gitdir=$(git rev-parse --absolute-git-dir)
branch=$(git symbolic-ref -q HEAD || echo refs/heads/main)
for remote in $(git remote); do git remote remove "$remote"; done
git update-ref --no-deref "$branch" "$base"
git symbolic-ref HEAD "$branch"
keep=$(git for-each-ref --merged "$base" --format='%(refname)' refs/tags; echo "$branch")
git for-each-ref --format='%(refname)' \
  | { grep -vxF -e "$keep" || true; } \
  | sed 's/^/delete /' | git update-ref --stdin
git worktree prune
rm -rf "$gitdir"/{logs,rr-cache,refs/original,rebase-merge,rebase-apply} \
  "$gitdir"/{ORIG_HEAD,FETCH_HEAD,MERGE_HEAD,CHERRY_PICK_HEAD,REVERT_HEAD,AUTO_MERGE} \
  "$gitdir"/info/refs "$gitdir"/objects/info/{packs,commit-graph,commit-graphs} \
  "$gitdir"/objects/pack/*.keep "$gitdir"/objects/pack/multi-pack-index
git -c gc.reflogExpire=now -c gc.reflogExpireUnreachable=now \
  -c gc.cruftPacks=false -c gc.writeCommitGraph=false gc -q --prune=now
test "$(git rev-parse HEAD)" = "$base"
test -z "$(git rev-list -n 1 --all --not "$base")"
"""


def base_commit(metadata: dict) -> str:
    base = metadata["base_commit"]
    if not re.fullmatch(r"[a-fA-F0-9]+", base):
        raise ValueError("invalid SWE base commit")
    return base


@solver
def base_history_only(repo: str = "/testbed") -> Solver:
    """Remove history after the sample's `base_commit` before the agent starts."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        result = await sandbox().exec(
            ["bash", "-c", HISTORY_SCRUB, "scrub", base_commit(state.metadata), repo],
            timeout=900,
        )
        if result.returncode:
            raise RuntimeError(
                f"could not remove history after the base commit: {result.stderr[-2000:]}"
            )
        return state

    return solve
