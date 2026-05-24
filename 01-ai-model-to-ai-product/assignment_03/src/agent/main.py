"""Interactive CLI: `uv run agent --session my_session`."""
from __future__ import annotations

import argparse

from agent.tools.dataset import load_bitext


def main() -> None:
    parser = argparse.ArgumentParser(description="Customer Service Data Analyst Agent")
    parser.add_argument("--session", default="default", help="Session ID for episodic memory")
    parser.add_argument("--user", default="default", help="User ID for profile memory")
    args = parser.parse_args()

    print(load_bitext())
    # TODO: build graph, loop over input(), stream reasoning steps + tool calls.
    print(f"[stub] starting session={args.session} user={args.user}")


if __name__ == "__main__":
    main()
